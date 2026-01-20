module c2f #(
  parameter int IN_CH = 1,
  parameter int OUT_CH = 1,
  parameter int MID_CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int N = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string CV1_WEIGHTS_FILE = "",
  parameter string CV1_BN_SCALE_FILE = "",
  parameter string CV1_BN_BIAS_FILE = "",
  parameter string CV2_WEIGHTS_FILE = "",
  parameter string CV2_BN_SCALE_FILE = "",
  parameter string CV2_BN_BIAS_FILE = "",
  parameter string BN0_CV1_WEIGHTS_FILE = "",
  parameter string BN0_CV1_BN_SCALE_FILE = "",
  parameter string BN0_CV1_BN_BIAS_FILE = "",
  parameter string BN0_CV2_WEIGHTS_FILE = "",
  parameter string BN0_CV2_BN_SCALE_FILE = "",
  parameter string BN0_CV2_BN_BIAS_FILE = "",
  parameter string BN1_CV1_WEIGHTS_FILE = "",
  parameter string BN1_CV1_BN_SCALE_FILE = "",
  parameter string BN1_CV1_BN_BIAS_FILE = "",
  parameter string BN1_CV2_WEIGHTS_FILE = "",
  parameter string BN1_CV2_BN_SCALE_FILE = "",
  parameter string BN1_CV2_BN_BIAS_FILE = ""
) (
  input  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] out_vec
);
  localparam int MAP_SIZE = IN_H*IN_W*WIDTH;
  localparam int SPLIT_CH = MID_CH;
  localparam int CAT_CH = (2 + N) * MID_CH;

  logic signed [2*MID_CH*IN_H*IN_W*WIDTH-1:0] cv1_out;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] x1;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] x2;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] bottleneck_out [0:N-1];
  logic signed [CAT_CH*IN_H*IN_W*WIDTH-1:0] cat_vec;

  genvar gi;

  yolo_conv #(
    .IN_CH(IN_CH),
    .OUT_CH(2*MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(1),
    .STRIDE(1),
    .PADDING(0),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(CV1_WEIGHTS_FILE),
    .BN_SCALE_FILE(CV1_BN_SCALE_FILE),
    .BN_BIAS_FILE(CV1_BN_BIAS_FILE)
  ) cv1 (
    .in_vec(in_vec),
    .out_vec(cv1_out)
  );

  assign x1 = cv1_out[0 +: MID_CH*MAP_SIZE];
  assign x2 = cv1_out[MID_CH*MAP_SIZE +: MID_CH*MAP_SIZE];

  generate
    if (N >= 1) begin : gen_bn0
      bottleneck #(
        .IN_CH(MID_CH),
        .OUT_CH(MID_CH),
        .IN_H(IN_H),
        .IN_W(IN_W),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .precision(precision),
        .SHORTCUT(1'b0),
        .CV1_WEIGHTS_FILE(BN0_CV1_WEIGHTS_FILE),
        .CV1_BN_SCALE_FILE(BN0_CV1_BN_SCALE_FILE),
        .CV1_BN_BIAS_FILE(BN0_CV1_BN_BIAS_FILE),
        .CV2_WEIGHTS_FILE(BN0_CV2_WEIGHTS_FILE),
        .CV2_BN_SCALE_FILE(BN0_CV2_BN_SCALE_FILE),
        .CV2_BN_BIAS_FILE(BN0_CV2_BN_BIAS_FILE)
      ) bn0 (
        .in_vec(x2),
        .out_vec(bottleneck_out[0])
      );
    end
    if (N >= 2) begin : gen_bn1
      bottleneck #(
        .IN_CH(MID_CH),
        .OUT_CH(MID_CH),
        .IN_H(IN_H),
        .IN_W(IN_W),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .precision(precision),
        .SHORTCUT(1'b0),
        .CV1_WEIGHTS_FILE(BN1_CV1_WEIGHTS_FILE),
        .CV1_BN_SCALE_FILE(BN1_CV1_BN_SCALE_FILE),
        .CV1_BN_BIAS_FILE(BN1_CV1_BN_BIAS_FILE),
        .CV2_WEIGHTS_FILE(BN1_CV2_WEIGHTS_FILE),
        .CV2_BN_SCALE_FILE(BN1_CV2_BN_SCALE_FILE),
        .CV2_BN_BIAS_FILE(BN1_CV2_BN_BIAS_FILE)
      ) bn1 (
        .in_vec(bottleneck_out[0]),
        .out_vec(bottleneck_out[1])
      );
    end
  endgenerate

  always_comb begin
    cat_vec = '0;
    cat_vec[0 +: MID_CH*MAP_SIZE] = x1;
    cat_vec[MID_CH*MAP_SIZE +: MID_CH*MAP_SIZE] = x2;
    for (int ci = 0; ci < N; ci = ci + 1) begin
      cat_vec[(2 + ci)*MID_CH*MAP_SIZE +: MID_CH*MAP_SIZE] = bottleneck_out[ci];
    end
  end

  yolo_conv #(
    .IN_CH(CAT_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(1),
    .STRIDE(1),
    .PADDING(0),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(CV2_WEIGHTS_FILE),
    .BN_SCALE_FILE(CV2_BN_SCALE_FILE),
    .BN_BIAS_FILE(CV2_BN_BIAS_FILE)
  ) cv2 (
    .in_vec(cat_vec),
    .out_vec(out_vec)
  );
endmodule
