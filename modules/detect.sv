module detect #(
  parameter int IN_CH1 = 1,
  parameter int IN_H1 = 1,
  parameter int IN_W1 = 1,
  parameter int IN_CH2 = 1,
  parameter int IN_H2 = 1,
  parameter int IN_W2 = 1,
  parameter int IN_CH3 = 1,
  parameter int IN_H3 = 1,
  parameter int IN_W3 = 1,
  parameter int REG_CH = 64,
  parameter int CLS_CH = 80,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string S1_R0_W = "",
  parameter string S1_R0_BN_S = "",
  parameter string S1_R0_BN_B = "",
  parameter string S1_R1_W = "",
  parameter string S1_R1_BN_S = "",
  parameter string S1_R1_BN_B = "",
  parameter string S1_R2_W = "",
  parameter string S1_R2_B = "",
  parameter string S1_C0_W = "",
  parameter string S1_C0_BN_S = "",
  parameter string S1_C0_BN_B = "",
  parameter string S1_C1_W = "",
  parameter string S1_C1_BN_S = "",
  parameter string S1_C1_BN_B = "",
  parameter string S1_C2_W = "",
  parameter string S1_C2_B = "",
  parameter string S2_R0_W = "",
  parameter string S2_R0_BN_S = "",
  parameter string S2_R0_BN_B = "",
  parameter string S2_R1_W = "",
  parameter string S2_R1_BN_S = "",
  parameter string S2_R1_BN_B = "",
  parameter string S2_R2_W = "",
  parameter string S2_R2_B = "",
  parameter string S2_C0_W = "",
  parameter string S2_C0_BN_S = "",
  parameter string S2_C0_BN_B = "",
  parameter string S2_C1_W = "",
  parameter string S2_C1_BN_S = "",
  parameter string S2_C1_BN_B = "",
  parameter string S2_C2_W = "",
  parameter string S2_C2_B = "",
  parameter string S3_R0_W = "",
  parameter string S3_R0_BN_S = "",
  parameter string S3_R0_BN_B = "",
  parameter string S3_R1_W = "",
  parameter string S3_R1_BN_S = "",
  parameter string S3_R1_BN_B = "",
  parameter string S3_R2_W = "",
  parameter string S3_R2_B = "",
  parameter string S3_C0_W = "",
  parameter string S3_C0_BN_S = "",
  parameter string S3_C0_BN_B = "",
  parameter string S3_C1_W = "",
  parameter string S3_C1_BN_S = "",
  parameter string S3_C1_BN_B = "",
  parameter string S3_C2_W = "",
  parameter string S3_C2_B = ""
) (
  input  logic signed [IN_CH1*IN_H1*IN_W1*WIDTH-1:0] in1,
  input  logic signed [IN_CH2*IN_H2*IN_W2*WIDTH-1:0] in2,
  input  logic signed [IN_CH3*IN_H3*IN_W3*WIDTH-1:0] in3,
  output logic signed [(REG_CH+CLS_CH)*(IN_H1*IN_W1 + IN_H2*IN_W2 + IN_H3*IN_W3)*WIDTH-1:0] out_vec
);
  localparam int OUT_CH = REG_CH + CLS_CH;

  logic signed [REG_CH*IN_H1*IN_W1*WIDTH-1:0] r1_0;
  logic signed [REG_CH*IN_H1*IN_W1*WIDTH-1:0] r1_1;
  logic signed [REG_CH*IN_H1*IN_W1*WIDTH-1:0] r1_out;
  logic signed [CLS_CH*IN_H1*IN_W1*WIDTH-1:0] c1_0;
  logic signed [CLS_CH*IN_H1*IN_W1*WIDTH-1:0] c1_1;
  logic signed [CLS_CH*IN_H1*IN_W1*WIDTH-1:0] c1_out;
  logic signed [OUT_CH*IN_H1*IN_W1*WIDTH-1:0] o1;

  logic signed [REG_CH*IN_H2*IN_W2*WIDTH-1:0] r2_0;
  logic signed [REG_CH*IN_H2*IN_W2*WIDTH-1:0] r2_1;
  logic signed [REG_CH*IN_H2*IN_W2*WIDTH-1:0] r2_out;
  logic signed [CLS_CH*IN_H2*IN_W2*WIDTH-1:0] c2_0;
  logic signed [CLS_CH*IN_H2*IN_W2*WIDTH-1:0] c2_1;
  logic signed [CLS_CH*IN_H2*IN_W2*WIDTH-1:0] c2_out;
  logic signed [OUT_CH*IN_H2*IN_W2*WIDTH-1:0] o2;

  logic signed [REG_CH*IN_H3*IN_W3*WIDTH-1:0] r3_0;
  logic signed [REG_CH*IN_H3*IN_W3*WIDTH-1:0] r3_1;
  logic signed [REG_CH*IN_H3*IN_W3*WIDTH-1:0] r3_out;
  logic signed [CLS_CH*IN_H3*IN_W3*WIDTH-1:0] c3_0;
  logic signed [CLS_CH*IN_H3*IN_W3*WIDTH-1:0] c3_1;
  logic signed [CLS_CH*IN_H3*IN_W3*WIDTH-1:0] c3_out;
  logic signed [OUT_CH*IN_H3*IN_W3*WIDTH-1:0] o3;

  yolo_conv #(
    .IN_CH(IN_CH1), .OUT_CH(REG_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_R0_W), .BN_SCALE_FILE(S1_R0_BN_S), .BN_BIAS_FILE(S1_R0_BN_B)
  ) s1_r0 (.in_vec(in1), .out_vec(r1_0));
  yolo_conv #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_R1_W), .BN_SCALE_FILE(S1_R1_BN_S), .BN_BIAS_FILE(S1_R1_BN_B)
  ) s1_r1 (.in_vec(r1_0), .out_vec(r1_1));
  conv2d #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_R2_W),
    .BIAS_FILE(S1_R2_B)
  ) s1_r2 (.in_vec(r1_1), .out_vec(r1_out));

  yolo_conv #(
    .IN_CH(IN_CH1), .OUT_CH(CLS_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_C0_W), .BN_SCALE_FILE(S1_C0_BN_S), .BN_BIAS_FILE(S1_C0_BN_B)
  ) s1_c0 (.in_vec(in1), .out_vec(c1_0));
  yolo_conv #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_C1_W), .BN_SCALE_FILE(S1_C1_BN_S), .BN_BIAS_FILE(S1_C1_BN_B)
  ) s1_c1 (.in_vec(c1_0), .out_vec(c1_1));
  conv2d #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H1), .IN_W(IN_W1),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S1_C2_W),
    .BIAS_FILE(S1_C2_B)
  ) s1_c2 (.in_vec(c1_1), .out_vec(c1_out));

  always_comb begin
    o1 = '0;
    o1[0 +: REG_CH*IN_H1*IN_W1*WIDTH] = r1_out;
    o1[REG_CH*IN_H1*IN_W1*WIDTH +: CLS_CH*IN_H1*IN_W1*WIDTH] = c1_out;
  end

  yolo_conv #(
    .IN_CH(IN_CH2), .OUT_CH(REG_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_R0_W), .BN_SCALE_FILE(S2_R0_BN_S), .BN_BIAS_FILE(S2_R0_BN_B)
  ) s2_r0 (.in_vec(in2), .out_vec(r2_0));
  yolo_conv #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_R1_W), .BN_SCALE_FILE(S2_R1_BN_S), .BN_BIAS_FILE(S2_R1_BN_B)
  ) s2_r1 (.in_vec(r2_0), .out_vec(r2_1));
  conv2d #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_R2_W),
    .BIAS_FILE(S2_R2_B)
  ) s2_r2 (.in_vec(r2_1), .out_vec(r2_out));

  yolo_conv #(
    .IN_CH(IN_CH2), .OUT_CH(CLS_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_C0_W), .BN_SCALE_FILE(S2_C0_BN_S), .BN_BIAS_FILE(S2_C0_BN_B)
  ) s2_c0 (.in_vec(in2), .out_vec(c2_0));
  yolo_conv #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_C1_W), .BN_SCALE_FILE(S2_C1_BN_S), .BN_BIAS_FILE(S2_C1_BN_B)
  ) s2_c1 (.in_vec(c2_0), .out_vec(c2_1));
  conv2d #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H2), .IN_W(IN_W2),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S2_C2_W),
    .BIAS_FILE(S2_C2_B)
  ) s2_c2 (.in_vec(c2_1), .out_vec(c2_out));

  always_comb begin
    o2 = '0;
    o2[0 +: REG_CH*IN_H2*IN_W2*WIDTH] = r2_out;
    o2[REG_CH*IN_H2*IN_W2*WIDTH +: CLS_CH*IN_H2*IN_W2*WIDTH] = c2_out;
  end

  yolo_conv #(
    .IN_CH(IN_CH3), .OUT_CH(REG_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_R0_W), .BN_SCALE_FILE(S3_R0_BN_S), .BN_BIAS_FILE(S3_R0_BN_B)
  ) s3_r0 (.in_vec(in3), .out_vec(r3_0));
  yolo_conv #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_R1_W), .BN_SCALE_FILE(S3_R1_BN_S), .BN_BIAS_FILE(S3_R1_BN_B)
  ) s3_r1 (.in_vec(r3_0), .out_vec(r3_1));
  conv2d #(
    .IN_CH(REG_CH), .OUT_CH(REG_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_R2_W),
    .BIAS_FILE(S3_R2_B)
  ) s3_r2 (.in_vec(r3_1), .out_vec(r3_out));

  yolo_conv #(
    .IN_CH(IN_CH3), .OUT_CH(CLS_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_C0_W), .BN_SCALE_FILE(S3_C0_BN_S), .BN_BIAS_FILE(S3_C0_BN_B)
  ) s3_c0 (.in_vec(in3), .out_vec(c3_0));
  yolo_conv #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(3), .STRIDE(1), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_C1_W), .BN_SCALE_FILE(S3_C1_BN_S), .BN_BIAS_FILE(S3_C1_BN_B)
  ) s3_c1 (.in_vec(c3_0), .out_vec(c3_1));
  conv2d #(
    .IN_CH(CLS_CH), .OUT_CH(CLS_CH), .IN_H(IN_H3), .IN_W(IN_W3),
    .K(1), .STRIDE(1), .PADDING(0), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(S3_C2_W),
    .BIAS_FILE(S3_C2_B)
  ) s3_c2 (.in_vec(c3_1), .out_vec(c3_out));

  always_comb begin
    o3 = '0;
    o3[0 +: REG_CH*IN_H3*IN_W3*WIDTH] = r3_out;
    o3[REG_CH*IN_H3*IN_W3*WIDTH +: CLS_CH*IN_H3*IN_W3*WIDTH] = c3_out;
  end

  always_comb begin
    out_vec = '0;
    out_vec[0 +: OUT_CH*IN_H1*IN_W1*WIDTH] = o1;
    out_vec[OUT_CH*IN_H1*IN_W1*WIDTH +: OUT_CH*IN_H2*IN_W2*WIDTH] = o2;
    out_vec[(OUT_CH*IN_H1*IN_W1 + OUT_CH*IN_H2*IN_W2)*WIDTH +: OUT_CH*IN_H3*IN_W3*WIDTH] = o3;
  end
endmodule
