module lenet5 #(
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string C1_WEIGHTS_FILE = "",
  parameter string C1_BIAS_FILE = "",
  parameter string C3_WEIGHTS_FILE = "",
  parameter string C3_BIAS_FILE = "",
  parameter string C5_WEIGHTS_FILE = "",
  parameter string C5_BIAS_FILE = "",
  parameter string F6_WEIGHTS_FILE = "",
  parameter string F6_BIAS_FILE = "",
  parameter string OUT_WEIGHTS_FILE = "",
  parameter string OUT_BIAS_FILE = ""
) (
  input  logic signed [1*32*32*WIDTH-1:0] in_vec,
  output logic signed [10*WIDTH-1:0] out_vec
);
  localparam int C1_OUT_H = 28;
  localparam int C1_OUT_W = 28;
  localparam int S2_OUT_H = 14;
  localparam int S2_OUT_W = 14;
  localparam int C3_OUT_H = 10;
  localparam int C3_OUT_W = 10;
  localparam int S4_OUT_H = 5;
  localparam int S4_OUT_W = 5;

  logic signed [6*C1_OUT_H*C1_OUT_W*WIDTH-1:0] c1_out;
  logic signed [6*C1_OUT_H*C1_OUT_W*WIDTH-1:0] c1_relu;
  logic signed [6*S2_OUT_H*S2_OUT_W*WIDTH-1:0] s2_out;
  logic signed [16*C3_OUT_H*C3_OUT_W*WIDTH-1:0] c3_out;
  logic signed [16*C3_OUT_H*C3_OUT_W*WIDTH-1:0] c3_relu;
  logic signed [16*S4_OUT_H*S4_OUT_W*WIDTH-1:0] s4_out;
  logic signed [120*WIDTH-1:0] c5_out;
  logic signed [120*WIDTH-1:0] c5_relu;
  logic signed [84*WIDTH-1:0] f6_out;
  logic signed [84*WIDTH-1:0] f6_relu;

  conv2d #(
    .IN_CH(1),
    .OUT_CH(6),
    .IN_H(32),
    .IN_W(32),
    .K(5),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(C1_WEIGHTS_FILE),
    .BIAS_FILE(C1_BIAS_FILE)
  ) c1 (
    .in_vec(in_vec),
    .out_vec(c1_out)
  );

  relu #(
    .DIM(6*C1_OUT_H*C1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu1 (
    .in_vec(c1_out),
    .out_vec(c1_relu)
  );

  avgpool2d #(
    .CH(6),
    .IN_H(C1_OUT_H),
    .IN_W(C1_OUT_W),
    .K(2),
    .STRIDE(2),
    .WIDTH(WIDTH),
    .precision(precision)
  ) s2 (
    .in_vec(c1_relu),
    .out_vec(s2_out)
  );

  conv2d #(
    .IN_CH(6),
    .OUT_CH(16),
    .IN_H(S2_OUT_H),
    .IN_W(S2_OUT_W),
    .K(5),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(C3_WEIGHTS_FILE),
    .BIAS_FILE(C3_BIAS_FILE)
  ) c3 (
    .in_vec(s2_out),
    .out_vec(c3_out)
  );

  relu #(
    .DIM(16*C3_OUT_H*C3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu2 (
    .in_vec(c3_out),
    .out_vec(c3_relu)
  );

  avgpool2d #(
    .CH(16),
    .IN_H(C3_OUT_H),
    .IN_W(C3_OUT_W),
    .K(2),
    .STRIDE(2),
    .WIDTH(WIDTH),
    .precision(precision)
  ) s4 (
    .in_vec(c3_relu),
    .out_vec(s4_out)
  );

  linear #(
    .IN_DIM(16*S4_OUT_H*S4_OUT_W),
    .OUT_DIM(120),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(C5_WEIGHTS_FILE),
    .BIAS_FILE(C5_BIAS_FILE)
  ) c5 (
    .in_vec(s4_out),
    .out_vec(c5_out)
  );

  relu #(
    .DIM(120),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu3 (
    .in_vec(c5_out),
    .out_vec(c5_relu)
  );

  linear #(
    .IN_DIM(120),
    .OUT_DIM(84),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(F6_WEIGHTS_FILE),
    .BIAS_FILE(F6_BIAS_FILE)
  ) f6 (
    .in_vec(c5_relu),
    .out_vec(f6_out)
  );

  relu #(
    .DIM(84),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu4 (
    .in_vec(f6_out),
    .out_vec(f6_relu)
  );

  linear #(
    .IN_DIM(84),
    .OUT_DIM(10),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(OUT_WEIGHTS_FILE),
    .BIAS_FILE(OUT_BIAS_FILE)
  ) out_layer (
    .in_vec(f6_relu),
    .out_vec(out_vec)
  );
endmodule
