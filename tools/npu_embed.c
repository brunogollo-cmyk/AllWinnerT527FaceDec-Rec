/* Read the SFace .nb output from the T527 NPU via VIPLite and print the
 * dequantised, L2-normalised 128-d embedding on stdout.
 *
 * This follows the exact call sequence vpm_run.c uses: load the network from
 * file, query the buffer descriptors the graph declares, create matching
 * buffers, set them, prepare, run, then read the output tensor. Doing it here
 * rather than through vpm_run is because vpm_run only writes a golden file when
 * the sample declares one, which makes numeric comparison awkward.
 *
 * Build:
 *   aarch64-linux-gnu-gcc -O2 -I<sdk>/inc -o npu_embed npu_embed.c \
 *       -L<sdk> -lVIPlite -lVIPuser -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "vip_lite.h"

#define DIM 128

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int build_buffer(vip_network network, int index, int is_input, vip_buffer *out)
{
    vip_buffer_create_params_t param;
    vip_char_t name[256];
    vip_status_e status;

    memset(&param, 0, sizeof(param));
    param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    memset(name, 0, sizeof(name));

    if (is_input) {
        vip_query_input(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
        vip_query_input(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
        vip_query_input(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
        vip_query_input(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
        vip_query_input(network, index, VIP_BUFFER_PROP_NAME, name);
        if (param.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
            vip_query_input(network, index, VIP_BUFFER_PROP_TF_SCALE, &param.quant_data.affine.scale);
            vip_query_input(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT, &param.quant_data.affine.zeroPoint);
        } else if (param.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
            vip_query_input(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &param.quant_data.dfp.fixed_point_pos);
        }
    } else {
        vip_query_output(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
        vip_query_output(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
        vip_query_output(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
        vip_query_output(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
        vip_query_output(network, index, VIP_BUFFER_PROP_NAME, name);
        if (param.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
            vip_query_output(network, index, VIP_BUFFER_PROP_TF_SCALE, &param.quant_data.affine.scale);
            vip_query_output(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT, &param.quant_data.affine.zeroPoint);
        } else if (param.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
            vip_query_output(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS, &param.quant_data.dfp.fixed_point_pos);
        }
    }

    fprintf(stderr, "%s %d: name=%s dims=%u shape=%ux%ux%u fmt=%d quant=%d scale=%.9f zp=%d\n",
            is_input ? "input" : "output", index, name, param.num_of_dims,
            param.sizes[0], param.sizes[1], param.sizes[2],
            (int)param.data_format, (int)param.quant_format,
            param.quant_data.affine.scale, param.quant_data.affine.zeroPoint);

    status = vip_create_buffer(&param, sizeof(param), out);
    if (status != VIP_SUCCESS)
        fprintf(stderr, "vip_create_buffer(%s %d) failed: %d\n",
                is_input ? "in" : "out", index, status);
    return status;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.nb> <input.dat> [loops]\n", argv[0]);
        return 2;
    }
    const char *nb_path = argv[1];
    const char *in_path = argv[2];
    int loops = (argc > 3) ? atoi(argv[3]) : 1;

    vip_status_e status;
    vip_network network = VIP_NULL;
    vip_buffer in_buf = VIP_NULL, out_buf = VIP_NULL;
    vip_uint32_t input_count = 0, output_count = 0, cid = 0;

    status = vip_init();
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_init failed: %d\n", status);
        return 1;
    }

    vip_query_hardware(VIP_QUERY_HW_PROP_CID, sizeof(vip_uint32_t), &cid);
    fprintf(stderr, "driver=0x%08x cid=0x%x\n", vip_get_version(), cid);

    status = vip_create_network(nb_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_create_network failed: %d\n", status);
        return 1;
    }

    vip_query_network(network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count);
    vip_query_network(network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count);
    fprintf(stderr, "inputs=%u outputs=%u\n", input_count, output_count);

    if (input_count < 1 || output_count < 1) {
        fprintf(stderr, "unexpected io counts\n");
        return 1;
    }
    if (build_buffer(network, 0, 1, &in_buf) != VIP_SUCCESS) return 1;
    if (build_buffer(network, 0, 0, &out_buf) != VIP_SUCCESS) return 1;

    vip_uint32_t in_size = vip_get_buffer_size(in_buf);
    void *in_map = vip_map_buffer(in_buf);
    FILE *f = fopen(in_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    size_t got = fread(in_map, 1, in_size, f);
    fclose(f);
    fprintf(stderr, "input buffer %u bytes, read %zu\n", in_size, got);
    vip_flush_buffer(in_buf, VIP_BUFFER_OPER_TYPE_FLUSH);
    vip_unmap_buffer(in_buf);

    double total = 0.0;
    for (int i = 0; i < loops; i++) {
        double t0 = now_ms();
        status = vip_prepare_network(network);
        if (status != VIP_SUCCESS) { fprintf(stderr, "prepare: %d\n", status); return 1; }
        status = vip_set_input(network, 0, in_buf);
        if (status != VIP_SUCCESS) { fprintf(stderr, "set_input: %d\n", status); return 1; }
        status = vip_set_output(network, 0, out_buf);
        if (status != VIP_SUCCESS) { fprintf(stderr, "set_output: %d\n", status); return 1; }
        status = vip_run_network(network);
        if (status != VIP_SUCCESS) { fprintf(stderr, "run: %d\n", status); return 1; }
        vip_finish_network(network);
        total += now_ms() - t0;
    }
    fprintf(stderr, "avg inference %.2f ms over %d loop(s)\n", total / loops, loops);

    vip_flush_buffer(out_buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
    vip_int8_t *out = (vip_int8_t *)vip_map_buffer(out_buf);

    float scale = 0.0f;
    vip_int32_t zero = 0;
    vip_query_output(network, 0, VIP_BUFFER_PROP_TF_SCALE, &scale);
    vip_query_output(network, 0, VIP_BUFFER_PROP_TF_ZERO_POINT, &zero);
    fprintf(stderr, "output scale=%.9f zero_point=%d\n", scale, zero);

    double norm = 0.0;
    float vals[DIM];
    for (int i = 0; i < DIM; i++) {
        vals[i] = ((float)out[i] - (float)zero) * scale;
        norm += (double)vals[i] * vals[i];
    }
    norm = norm > 0.0 ? 1.0 / sqrt(norm) : 0.0;

    for (int i = 0; i < DIM; i++)
        printf("%.6f%s", vals[i] * (float)norm, i == DIM - 1 ? "\n" : ",");

    vip_unmap_buffer(out_buf);
    vip_destroy_buffer(in_buf);
    vip_destroy_buffer(out_buf);
    vip_destroy_network(network);
    return 0;
}
