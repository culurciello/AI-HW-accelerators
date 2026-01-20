module yolo_conv #(
  parameter int IN_CH = 1,
  parameter int OUT_CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int K = 3,
  parameter int STRIDE = 1,
  parameter int PADDING = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string WEIGHTS_FILE = "",
  parameter string BN_SCALE_FILE = "",
  parameter string BN_BIAS_FILE = ""
) (
  input  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [OUT_CH*((IN_H+2*PADDING-K)/STRIDE+1)*((IN_W+2*PADDING-K)/STRIDE+1)*WIDTH-1:0] out_vec
);
  localparam int OUT_H = (IN_H + 2*PADDING - K) / STRIDE + 1;
  localparam int OUT_W = (IN_W + 2*PADDING - K) / STRIDE + 1;

  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] conv_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] bn_out;

  conv2d #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(K),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(WEIGHTS_FILE)
  ) conv (
    .in_vec(in_vec),
    .out_vec(conv_out)
  );

  batchnorm2d #(
    .CH(OUT_CH),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(BN_SCALE_FILE),
    .BIAS_FILE(BN_BIAS_FILE)
  ) bn (
    .in_vec(conv_out),
    .out_vec(bn_out)
  );

  silu #(
    .DIM(OUT_CH*OUT_H*OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision)
  ) act (
    .in_vec(bn_out),
    .out_vec(out_vec)
  );
endmodule
