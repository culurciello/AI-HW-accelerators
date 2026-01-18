module mlp_c1 #(
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string L1_WEIGHTS_FILE = "",
  parameter string L2_WEIGHTS_FILE = "",
  parameter string L3_WEIGHTS_FILE = ""
) (
  input  logic signed [10*WIDTH-1:0] in_vec,
  output logic signed [2*WIDTH-1:0] out_vec
);
  logic signed [32*WIDTH-1:0] l1_out;
  logic signed [32*WIDTH-1:0] l1_relu;
  logic signed [32*WIDTH-1:0] l2_out;
  logic signed [32*WIDTH-1:0] l2_relu;

  linear #(
    .IN_DIM(10),
    .OUT_DIM(32),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L1_WEIGHTS_FILE)
  ) layer1 (
    .in_vec(in_vec),
    .out_vec(l1_out)
  );

  relu #(
    .DIM(32),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu1 (
    .in_vec(l1_out),
    .out_vec(l1_relu)
  );

  linear #(
    .IN_DIM(32),
    .OUT_DIM(32),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_WEIGHTS_FILE)
  ) layer2 (
    .in_vec(l1_relu),
    .out_vec(l2_out)
  );

  relu #(
    .DIM(32),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu2 (
    .in_vec(l2_out),
    .out_vec(l2_relu)
  );

  linear #(
    .IN_DIM(32),
    .OUT_DIM(2),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_WEIGHTS_FILE)
  ) layer3 (
    .in_vec(l2_relu),
    .out_vec(out_vec)
  );
endmodule
