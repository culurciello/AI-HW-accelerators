module yolov8n #(
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter int IN_H = 640,
  parameter int IN_W = 640,
  parameter string CV0_W = "",
  parameter string CV0_BN_S = "",
  parameter string CV0_BN_B = "",
  parameter string CV1_W = "",
  parameter string CV1_BN_S = "",
  parameter string CV1_BN_B = "",
  parameter string C2_CV1_W = "",
  parameter string C2_CV1_BN_S = "",
  parameter string C2_CV1_BN_B = "",
  parameter string C2_CV2_W = "",
  parameter string C2_CV2_BN_S = "",
  parameter string C2_CV2_BN_B = "",
  parameter string C2_BN0_CV1_W = "",
  parameter string C2_BN0_CV1_BN_S = "",
  parameter string C2_BN0_CV1_BN_B = "",
  parameter string C2_BN0_CV2_W = "",
  parameter string C2_BN0_CV2_BN_S = "",
  parameter string C2_BN0_CV2_BN_B = "",
  parameter string CV3_W = "",
  parameter string CV3_BN_S = "",
  parameter string CV3_BN_B = "",
  parameter string C4_CV1_W = "",
  parameter string C4_CV1_BN_S = "",
  parameter string C4_CV1_BN_B = "",
  parameter string C4_CV2_W = "",
  parameter string C4_CV2_BN_S = "",
  parameter string C4_CV2_BN_B = "",
  parameter string C4_BN0_CV1_W = "",
  parameter string C4_BN0_CV1_BN_S = "",
  parameter string C4_BN0_CV1_BN_B = "",
  parameter string C4_BN0_CV2_W = "",
  parameter string C4_BN0_CV2_BN_S = "",
  parameter string C4_BN0_CV2_BN_B = "",
  parameter string C4_BN1_CV1_W = "",
  parameter string C4_BN1_CV1_BN_S = "",
  parameter string C4_BN1_CV1_BN_B = "",
  parameter string C4_BN1_CV2_W = "",
  parameter string C4_BN1_CV2_BN_S = "",
  parameter string C4_BN1_CV2_BN_B = "",
  parameter string CV5_W = "",
  parameter string CV5_BN_S = "",
  parameter string CV5_BN_B = "",
  parameter string C6_CV1_W = "",
  parameter string C6_CV1_BN_S = "",
  parameter string C6_CV1_BN_B = "",
  parameter string C6_CV2_W = "",
  parameter string C6_CV2_BN_S = "",
  parameter string C6_CV2_BN_B = "",
  parameter string C6_BN0_CV1_W = "",
  parameter string C6_BN0_CV1_BN_S = "",
  parameter string C6_BN0_CV1_BN_B = "",
  parameter string C6_BN0_CV2_W = "",
  parameter string C6_BN0_CV2_BN_S = "",
  parameter string C6_BN0_CV2_BN_B = "",
  parameter string C6_BN1_CV1_W = "",
  parameter string C6_BN1_CV1_BN_S = "",
  parameter string C6_BN1_CV1_BN_B = "",
  parameter string C6_BN1_CV2_W = "",
  parameter string C6_BN1_CV2_BN_S = "",
  parameter string C6_BN1_CV2_BN_B = "",
  parameter string CV7_W = "",
  parameter string CV7_BN_S = "",
  parameter string CV7_BN_B = "",
  parameter string C8_CV1_W = "",
  parameter string C8_CV1_BN_S = "",
  parameter string C8_CV1_BN_B = "",
  parameter string C8_CV2_W = "",
  parameter string C8_CV2_BN_S = "",
  parameter string C8_CV2_BN_B = "",
  parameter string C8_BN0_CV1_W = "",
  parameter string C8_BN0_CV1_BN_S = "",
  parameter string C8_BN0_CV1_BN_B = "",
  parameter string C8_BN0_CV2_W = "",
  parameter string C8_BN0_CV2_BN_S = "",
  parameter string C8_BN0_CV2_BN_B = "",
  parameter string SPPF_CV1_W = "",
  parameter string SPPF_CV1_BN_S = "",
  parameter string SPPF_CV1_BN_B = "",
  parameter string SPPF_CV2_W = "",
  parameter string SPPF_CV2_BN_S = "",
  parameter string SPPF_CV2_BN_B = "",
  parameter string C12_CV1_W = "",
  parameter string C12_CV1_BN_S = "",
  parameter string C12_CV1_BN_B = "",
  parameter string C12_CV2_W = "",
  parameter string C12_CV2_BN_S = "",
  parameter string C12_CV2_BN_B = "",
  parameter string C12_BN0_CV1_W = "",
  parameter string C12_BN0_CV1_BN_S = "",
  parameter string C12_BN0_CV1_BN_B = "",
  parameter string C12_BN0_CV2_W = "",
  parameter string C12_BN0_CV2_BN_S = "",
  parameter string C12_BN0_CV2_BN_B = "",
  parameter string C15_CV1_W = "",
  parameter string C15_CV1_BN_S = "",
  parameter string C15_CV1_BN_B = "",
  parameter string C15_CV2_W = "",
  parameter string C15_CV2_BN_S = "",
  parameter string C15_CV2_BN_B = "",
  parameter string C15_BN0_CV1_W = "",
  parameter string C15_BN0_CV1_BN_S = "",
  parameter string C15_BN0_CV1_BN_B = "",
  parameter string C15_BN0_CV2_W = "",
  parameter string C15_BN0_CV2_BN_S = "",
  parameter string C15_BN0_CV2_BN_B = "",
  parameter string CV16_W = "",
  parameter string CV16_BN_S = "",
  parameter string CV16_BN_B = "",
  parameter string C18_CV1_W = "",
  parameter string C18_CV1_BN_S = "",
  parameter string C18_CV1_BN_B = "",
  parameter string C18_CV2_W = "",
  parameter string C18_CV2_BN_S = "",
  parameter string C18_CV2_BN_B = "",
  parameter string C18_BN0_CV1_W = "",
  parameter string C18_BN0_CV1_BN_S = "",
  parameter string C18_BN0_CV1_BN_B = "",
  parameter string C18_BN0_CV2_W = "",
  parameter string C18_BN0_CV2_BN_S = "",
  parameter string C18_BN0_CV2_BN_B = "",
  parameter string CV19_W = "",
  parameter string CV19_BN_S = "",
  parameter string CV19_BN_B = "",
  parameter string C21_CV1_W = "",
  parameter string C21_CV1_BN_S = "",
  parameter string C21_CV1_BN_B = "",
  parameter string C21_CV2_W = "",
  parameter string C21_CV2_BN_S = "",
  parameter string C21_CV2_BN_B = "",
  parameter string C21_BN0_CV1_W = "",
  parameter string C21_BN0_CV1_BN_S = "",
  parameter string C21_BN0_CV1_BN_B = "",
  parameter string C21_BN0_CV2_W = "",
  parameter string C21_BN0_CV2_BN_S = "",
  parameter string C21_BN0_CV2_BN_B = "",
  parameter string D_S1_R0_W = "",
  parameter string D_S1_R0_BN_S = "",
  parameter string D_S1_R0_BN_B = "",
  parameter string D_S1_R1_W = "",
  parameter string D_S1_R1_BN_S = "",
  parameter string D_S1_R1_BN_B = "",
  parameter string D_S1_R2_W = "",
  parameter string D_S1_R2_B = "",
  parameter string D_S1_C0_W = "",
  parameter string D_S1_C0_BN_S = "",
  parameter string D_S1_C0_BN_B = "",
  parameter string D_S1_C1_W = "",
  parameter string D_S1_C1_BN_S = "",
  parameter string D_S1_C1_BN_B = "",
  parameter string D_S1_C2_W = "",
  parameter string D_S1_C2_B = "",
  parameter string D_S2_R0_W = "",
  parameter string D_S2_R0_BN_S = "",
  parameter string D_S2_R0_BN_B = "",
  parameter string D_S2_R1_W = "",
  parameter string D_S2_R1_BN_S = "",
  parameter string D_S2_R1_BN_B = "",
  parameter string D_S2_R2_W = "",
  parameter string D_S2_R2_B = "",
  parameter string D_S2_C0_W = "",
  parameter string D_S2_C0_BN_S = "",
  parameter string D_S2_C0_BN_B = "",
  parameter string D_S2_C1_W = "",
  parameter string D_S2_C1_BN_S = "",
  parameter string D_S2_C1_BN_B = "",
  parameter string D_S2_C2_W = "",
  parameter string D_S2_C2_B = "",
  parameter string D_S3_R0_W = "",
  parameter string D_S3_R0_BN_S = "",
  parameter string D_S3_R0_BN_B = "",
  parameter string D_S3_R1_W = "",
  parameter string D_S3_R1_BN_S = "",
  parameter string D_S3_R1_BN_B = "",
  parameter string D_S3_R2_W = "",
  parameter string D_S3_R2_B = "",
  parameter string D_S3_C0_W = "",
  parameter string D_S3_C0_BN_S = "",
  parameter string D_S3_C0_BN_B = "",
  parameter string D_S3_C1_W = "",
  parameter string D_S3_C1_BN_S = "",
  parameter string D_S3_C1_BN_B = "",
  parameter string D_S3_C2_W = "",
  parameter string D_S3_C2_B = ""
) (
  input  logic signed [3*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [(64+80)*((IN_H/8)*(IN_W/8) + (IN_H/16)*(IN_W/16) + (IN_H/32)*(IN_W/32))*WIDTH-1:0] out_vec
);
  localparam int H1 = (IN_H + 1) / 2;
  localparam int W1 = (IN_W + 1) / 2;
  localparam int H2 = (H1 + 1) / 2;
  localparam int W2 = (W1 + 1) / 2;
  localparam int H3 = (H2 + 1) / 2;
  localparam int W3 = (W2 + 1) / 2;
  localparam int H4 = (H3 + 1) / 2;
  localparam int W4 = (W3 + 1) / 2;
  localparam int H5 = (H4 + 1) / 2;
  localparam int W5 = (W4 + 1) / 2;

  logic signed [16*H1*W1*WIDTH-1:0] x0;
  logic signed [32*H2*W2*WIDTH-1:0] x1;
  logic signed [32*H2*W2*WIDTH-1:0] x2;
  logic signed [64*H3*W3*WIDTH-1:0] x3;
  logic signed [64*H3*W3*WIDTH-1:0] x4;
  logic signed [128*H4*W4*WIDTH-1:0] x5;
  logic signed [128*H4*W4*WIDTH-1:0] x6;
  logic signed [256*H5*W5*WIDTH-1:0] x7;
  logic signed [256*H5*W5*WIDTH-1:0] x8;
  logic signed [256*H5*W5*WIDTH-1:0] x9;
  logic signed [256*H4*W4*WIDTH-1:0] x10;
  logic signed [384*H4*W4*WIDTH-1:0] x11;
  logic signed [128*H4*W4*WIDTH-1:0] x12;
  logic signed [128*H3*W3*WIDTH-1:0] x13;
  logic signed [192*H3*W3*WIDTH-1:0] x14;
  logic signed [64*H3*W3*WIDTH-1:0] x15;
  logic signed [64*H4*W4*WIDTH-1:0] x16;
  logic signed [192*H4*W4*WIDTH-1:0] x17;
  logic signed [128*H4*W4*WIDTH-1:0] x18;
  logic signed [128*H5*W5*WIDTH-1:0] x19;
  logic signed [384*H5*W5*WIDTH-1:0] x20;
  logic signed [256*H5*W5*WIDTH-1:0] x21;

  yolo_conv #(.IN_CH(3), .OUT_CH(16), .IN_H(IN_H), .IN_W(IN_W), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV0_W), .BN_SCALE_FILE(CV0_BN_S), .BN_BIAS_FILE(CV0_BN_B)) cv0 (.in_vec(in_vec), .out_vec(x0));
  yolo_conv #(.IN_CH(16), .OUT_CH(32), .IN_H(H1), .IN_W(W1), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV1_W), .BN_SCALE_FILE(CV1_BN_S), .BN_BIAS_FILE(CV1_BN_B)) cv1 (.in_vec(x0), .out_vec(x1));
  c2f #(.IN_CH(32), .OUT_CH(32), .MID_CH(16), .IN_H(H2), .IN_W(W2), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C2_CV1_W), .CV1_BN_SCALE_FILE(C2_CV1_BN_S), .CV1_BN_BIAS_FILE(C2_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C2_CV2_W), .CV2_BN_SCALE_FILE(C2_CV2_BN_S), .CV2_BN_BIAS_FILE(C2_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C2_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C2_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C2_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C2_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C2_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C2_BN0_CV2_BN_B)) c2 (.in_vec(x1), .out_vec(x2));
  yolo_conv #(.IN_CH(32), .OUT_CH(64), .IN_H(H2), .IN_W(W2), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV3_W), .BN_SCALE_FILE(CV3_BN_S), .BN_BIAS_FILE(CV3_BN_B)) cv3 (.in_vec(x2), .out_vec(x3));
  c2f #(.IN_CH(64), .OUT_CH(64), .MID_CH(32), .IN_H(H3), .IN_W(W3), .N(2), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C4_CV1_W), .CV1_BN_SCALE_FILE(C4_CV1_BN_S), .CV1_BN_BIAS_FILE(C4_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C4_CV2_W), .CV2_BN_SCALE_FILE(C4_CV2_BN_S), .CV2_BN_BIAS_FILE(C4_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C4_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C4_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C4_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C4_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C4_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C4_BN0_CV2_BN_B),
    .BN1_CV1_WEIGHTS_FILE(C4_BN1_CV1_W), .BN1_CV1_BN_SCALE_FILE(C4_BN1_CV1_BN_S), .BN1_CV1_BN_BIAS_FILE(C4_BN1_CV1_BN_B),
    .BN1_CV2_WEIGHTS_FILE(C4_BN1_CV2_W), .BN1_CV2_BN_SCALE_FILE(C4_BN1_CV2_BN_S), .BN1_CV2_BN_BIAS_FILE(C4_BN1_CV2_BN_B)) c4 (.in_vec(x3), .out_vec(x4));
  yolo_conv #(.IN_CH(64), .OUT_CH(128), .IN_H(H3), .IN_W(W3), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV5_W), .BN_SCALE_FILE(CV5_BN_S), .BN_BIAS_FILE(CV5_BN_B)) cv5 (.in_vec(x4), .out_vec(x5));
  c2f #(.IN_CH(128), .OUT_CH(128), .MID_CH(64), .IN_H(H4), .IN_W(W4), .N(2), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C6_CV1_W), .CV1_BN_SCALE_FILE(C6_CV1_BN_S), .CV1_BN_BIAS_FILE(C6_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C6_CV2_W), .CV2_BN_SCALE_FILE(C6_CV2_BN_S), .CV2_BN_BIAS_FILE(C6_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C6_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C6_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C6_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C6_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C6_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C6_BN0_CV2_BN_B),
    .BN1_CV1_WEIGHTS_FILE(C6_BN1_CV1_W), .BN1_CV1_BN_SCALE_FILE(C6_BN1_CV1_BN_S), .BN1_CV1_BN_BIAS_FILE(C6_BN1_CV1_BN_B),
    .BN1_CV2_WEIGHTS_FILE(C6_BN1_CV2_W), .BN1_CV2_BN_SCALE_FILE(C6_BN1_CV2_BN_S), .BN1_CV2_BN_BIAS_FILE(C6_BN1_CV2_BN_B)) c6 (.in_vec(x5), .out_vec(x6));
  yolo_conv #(.IN_CH(128), .OUT_CH(256), .IN_H(H4), .IN_W(W4), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV7_W), .BN_SCALE_FILE(CV7_BN_S), .BN_BIAS_FILE(CV7_BN_B)) cv7 (.in_vec(x6), .out_vec(x7));
  c2f #(.IN_CH(256), .OUT_CH(256), .MID_CH(128), .IN_H(H5), .IN_W(W5), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C8_CV1_W), .CV1_BN_SCALE_FILE(C8_CV1_BN_S), .CV1_BN_BIAS_FILE(C8_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C8_CV2_W), .CV2_BN_SCALE_FILE(C8_CV2_BN_S), .CV2_BN_BIAS_FILE(C8_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C8_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C8_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C8_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C8_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C8_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C8_BN0_CV2_BN_B)) c8 (.in_vec(x7), .out_vec(x8));
  sppf #(.IN_CH(256), .OUT_CH(256), .MID_CH(128), .IN_H(H5), .IN_W(W5), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(SPPF_CV1_W), .CV1_BN_SCALE_FILE(SPPF_CV1_BN_S), .CV1_BN_BIAS_FILE(SPPF_CV1_BN_B),
    .CV2_WEIGHTS_FILE(SPPF_CV2_W), .CV2_BN_SCALE_FILE(SPPF_CV2_BN_S), .CV2_BN_BIAS_FILE(SPPF_CV2_BN_B)) sppf0 (.in_vec(x8), .out_vec(x9));
  upsample2d #(.CH(256), .IN_H(H5), .IN_W(W5), .SCALE(2), .WIDTH(WIDTH), .precision(precision)) up10 (.in_vec(x9), .out_vec(x10));
  concat2d #(.A_CH(256), .B_CH(128), .IN_H(H4), .IN_W(W4), .WIDTH(WIDTH), .precision(precision)) cat11 (.a_vec(x10), .b_vec(x6), .out_vec(x11));
  c2f #(.IN_CH(384), .OUT_CH(128), .MID_CH(64), .IN_H(H4), .IN_W(W4), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C12_CV1_W), .CV1_BN_SCALE_FILE(C12_CV1_BN_S), .CV1_BN_BIAS_FILE(C12_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C12_CV2_W), .CV2_BN_SCALE_FILE(C12_CV2_BN_S), .CV2_BN_BIAS_FILE(C12_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C12_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C12_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C12_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C12_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C12_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C12_BN0_CV2_BN_B)) c12 (.in_vec(x11), .out_vec(x12));
  upsample2d #(.CH(128), .IN_H(H4), .IN_W(W4), .SCALE(2), .WIDTH(WIDTH), .precision(precision)) up13 (.in_vec(x12), .out_vec(x13));
  concat2d #(.A_CH(128), .B_CH(64), .IN_H(H3), .IN_W(W3), .WIDTH(WIDTH), .precision(precision)) cat14 (.a_vec(x13), .b_vec(x4), .out_vec(x14));
  c2f #(.IN_CH(192), .OUT_CH(64), .MID_CH(32), .IN_H(H3), .IN_W(W3), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C15_CV1_W), .CV1_BN_SCALE_FILE(C15_CV1_BN_S), .CV1_BN_BIAS_FILE(C15_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C15_CV2_W), .CV2_BN_SCALE_FILE(C15_CV2_BN_S), .CV2_BN_BIAS_FILE(C15_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C15_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C15_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C15_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C15_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C15_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C15_BN0_CV2_BN_B)) c15 (.in_vec(x14), .out_vec(x15));
  yolo_conv #(.IN_CH(64), .OUT_CH(64), .IN_H(H3), .IN_W(W3), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV16_W), .BN_SCALE_FILE(CV16_BN_S), .BN_BIAS_FILE(CV16_BN_B)) cv16 (.in_vec(x15), .out_vec(x16));
  concat2d #(.A_CH(64), .B_CH(128), .IN_H(H4), .IN_W(W4), .WIDTH(WIDTH), .precision(precision)) cat17 (.a_vec(x16), .b_vec(x12), .out_vec(x17));
  c2f #(.IN_CH(192), .OUT_CH(128), .MID_CH(64), .IN_H(H4), .IN_W(W4), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C18_CV1_W), .CV1_BN_SCALE_FILE(C18_CV1_BN_S), .CV1_BN_BIAS_FILE(C18_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C18_CV2_W), .CV2_BN_SCALE_FILE(C18_CV2_BN_S), .CV2_BN_BIAS_FILE(C18_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C18_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C18_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C18_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C18_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C18_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C18_BN0_CV2_BN_B)) c18 (.in_vec(x17), .out_vec(x18));
  yolo_conv #(.IN_CH(128), .OUT_CH(128), .IN_H(H4), .IN_W(W4), .K(3), .STRIDE(2), .PADDING(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .WEIGHTS_FILE(CV19_W), .BN_SCALE_FILE(CV19_BN_S), .BN_BIAS_FILE(CV19_BN_B)) cv19 (.in_vec(x18), .out_vec(x19));
  concat2d #(.A_CH(128), .B_CH(256), .IN_H(H5), .IN_W(W5), .WIDTH(WIDTH), .precision(precision)) cat20 (.a_vec(x19), .b_vec(x9), .out_vec(x20));
  c2f #(.IN_CH(384), .OUT_CH(256), .MID_CH(128), .IN_H(H5), .IN_W(W5), .N(1), .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .CV1_WEIGHTS_FILE(C21_CV1_W), .CV1_BN_SCALE_FILE(C21_CV1_BN_S), .CV1_BN_BIAS_FILE(C21_CV1_BN_B),
    .CV2_WEIGHTS_FILE(C21_CV2_W), .CV2_BN_SCALE_FILE(C21_CV2_BN_S), .CV2_BN_BIAS_FILE(C21_CV2_BN_B),
    .BN0_CV1_WEIGHTS_FILE(C21_BN0_CV1_W), .BN0_CV1_BN_SCALE_FILE(C21_BN0_CV1_BN_S), .BN0_CV1_BN_BIAS_FILE(C21_BN0_CV1_BN_B),
    .BN0_CV2_WEIGHTS_FILE(C21_BN0_CV2_W), .BN0_CV2_BN_SCALE_FILE(C21_BN0_CV2_BN_S), .BN0_CV2_BN_BIAS_FILE(C21_BN0_CV2_BN_B)) c21 (.in_vec(x20), .out_vec(x21));

  detect #(
    .IN_CH1(64), .IN_H1(H3), .IN_W1(W3),
    .IN_CH2(128), .IN_H2(H4), .IN_W2(W4),
    .IN_CH3(256), .IN_H3(H5), .IN_W3(W5),
    .REG_CH(64), .CLS_CH(80),
    .WIDTH(WIDTH), .FRAC(FRAC), .precision(precision),
    .S1_R0_W(D_S1_R0_W), .S1_R0_BN_S(D_S1_R0_BN_S), .S1_R0_BN_B(D_S1_R0_BN_B),
    .S1_R1_W(D_S1_R1_W), .S1_R1_BN_S(D_S1_R1_BN_S), .S1_R1_BN_B(D_S1_R1_BN_B),
    .S1_R2_W(D_S1_R2_W), .S1_R2_B(D_S1_R2_B),
    .S1_C0_W(D_S1_C0_W), .S1_C0_BN_S(D_S1_C0_BN_S), .S1_C0_BN_B(D_S1_C0_BN_B),
    .S1_C1_W(D_S1_C1_W), .S1_C1_BN_S(D_S1_C1_BN_S), .S1_C1_BN_B(D_S1_C1_BN_B),
    .S1_C2_W(D_S1_C2_W), .S1_C2_B(D_S1_C2_B),
    .S2_R0_W(D_S2_R0_W), .S2_R0_BN_S(D_S2_R0_BN_S), .S2_R0_BN_B(D_S2_R0_BN_B),
    .S2_R1_W(D_S2_R1_W), .S2_R1_BN_S(D_S2_R1_BN_S), .S2_R1_BN_B(D_S2_R1_BN_B),
    .S2_R2_W(D_S2_R2_W), .S2_R2_B(D_S2_R2_B),
    .S2_C0_W(D_S2_C0_W), .S2_C0_BN_S(D_S2_C0_BN_S), .S2_C0_BN_B(D_S2_C0_BN_B),
    .S2_C1_W(D_S2_C1_W), .S2_C1_BN_S(D_S2_C1_BN_S), .S2_C1_BN_B(D_S2_C1_BN_B),
    .S2_C2_W(D_S2_C2_W), .S2_C2_B(D_S2_C2_B),
    .S3_R0_W(D_S3_R0_W), .S3_R0_BN_S(D_S3_R0_BN_S), .S3_R0_BN_B(D_S3_R0_BN_B),
    .S3_R1_W(D_S3_R1_W), .S3_R1_BN_S(D_S3_R1_BN_S), .S3_R1_BN_B(D_S3_R1_BN_B),
    .S3_R2_W(D_S3_R2_W), .S3_R2_B(D_S3_R2_B),
    .S3_C0_W(D_S3_C0_W), .S3_C0_BN_S(D_S3_C0_BN_S), .S3_C0_BN_B(D_S3_C0_BN_B),
    .S3_C1_W(D_S3_C1_W), .S3_C1_BN_S(D_S3_C1_BN_S), .S3_C1_BN_B(D_S3_C1_BN_B),
    .S3_C2_W(D_S3_C2_W), .S3_C2_B(D_S3_C2_B)
  ) det (
    .in1(x15),
    .in2(x18),
    .in3(x21),
    .out_vec(out_vec)
  );
endmodule
