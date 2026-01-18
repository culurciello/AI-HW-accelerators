module resnet18 #(
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string CONV1_WEIGHTS_FILE = "",
  parameter string BN1_SCALE_FILE = "",
  parameter string BN1_BIAS_FILE = "",
  parameter string L1_0_CONV1_WEIGHTS_FILE = "",
  parameter string L1_0_BN1_SCALE_FILE = "",
  parameter string L1_0_BN1_BIAS_FILE = "",
  parameter string L1_0_CONV2_WEIGHTS_FILE = "",
  parameter string L1_0_BN2_SCALE_FILE = "",
  parameter string L1_0_BN2_BIAS_FILE = "",
  parameter string L1_1_CONV1_WEIGHTS_FILE = "",
  parameter string L1_1_BN1_SCALE_FILE = "",
  parameter string L1_1_BN1_BIAS_FILE = "",
  parameter string L1_1_CONV2_WEIGHTS_FILE = "",
  parameter string L1_1_BN2_SCALE_FILE = "",
  parameter string L1_1_BN2_BIAS_FILE = "",
  parameter string L2_0_CONV1_WEIGHTS_FILE = "",
  parameter string L2_0_BN1_SCALE_FILE = "",
  parameter string L2_0_BN1_BIAS_FILE = "",
  parameter string L2_0_CONV2_WEIGHTS_FILE = "",
  parameter string L2_0_BN2_SCALE_FILE = "",
  parameter string L2_0_BN2_BIAS_FILE = "",
  parameter string L2_0_DS_CONV_WEIGHTS_FILE = "",
  parameter string L2_0_DS_BN_SCALE_FILE = "",
  parameter string L2_0_DS_BN_BIAS_FILE = "",
  parameter string L2_1_CONV1_WEIGHTS_FILE = "",
  parameter string L2_1_BN1_SCALE_FILE = "",
  parameter string L2_1_BN1_BIAS_FILE = "",
  parameter string L2_1_CONV2_WEIGHTS_FILE = "",
  parameter string L2_1_BN2_SCALE_FILE = "",
  parameter string L2_1_BN2_BIAS_FILE = "",
  parameter string L3_0_CONV1_WEIGHTS_FILE = "",
  parameter string L3_0_BN1_SCALE_FILE = "",
  parameter string L3_0_BN1_BIAS_FILE = "",
  parameter string L3_0_CONV2_WEIGHTS_FILE = "",
  parameter string L3_0_BN2_SCALE_FILE = "",
  parameter string L3_0_BN2_BIAS_FILE = "",
  parameter string L3_0_DS_CONV_WEIGHTS_FILE = "",
  parameter string L3_0_DS_BN_SCALE_FILE = "",
  parameter string L3_0_DS_BN_BIAS_FILE = "",
  parameter string L3_1_CONV1_WEIGHTS_FILE = "",
  parameter string L3_1_BN1_SCALE_FILE = "",
  parameter string L3_1_BN1_BIAS_FILE = "",
  parameter string L3_1_CONV2_WEIGHTS_FILE = "",
  parameter string L3_1_BN2_SCALE_FILE = "",
  parameter string L3_1_BN2_BIAS_FILE = "",
  parameter string L4_0_CONV1_WEIGHTS_FILE = "",
  parameter string L4_0_BN1_SCALE_FILE = "",
  parameter string L4_0_BN1_BIAS_FILE = "",
  parameter string L4_0_CONV2_WEIGHTS_FILE = "",
  parameter string L4_0_BN2_SCALE_FILE = "",
  parameter string L4_0_BN2_BIAS_FILE = "",
  parameter string L4_0_DS_CONV_WEIGHTS_FILE = "",
  parameter string L4_0_DS_BN_SCALE_FILE = "",
  parameter string L4_0_DS_BN_BIAS_FILE = "",
  parameter string L4_1_CONV1_WEIGHTS_FILE = "",
  parameter string L4_1_BN1_SCALE_FILE = "",
  parameter string L4_1_BN1_BIAS_FILE = "",
  parameter string L4_1_CONV2_WEIGHTS_FILE = "",
  parameter string L4_1_BN2_SCALE_FILE = "",
  parameter string L4_1_BN2_BIAS_FILE = "",
  parameter string FC_WEIGHTS_FILE = "",
  parameter string FC_BIAS_FILE = ""
) (
  input  logic signed [3*224*224*WIDTH-1:0] in_vec,
  output logic signed [1000*WIDTH-1:0] out_vec
);
  localparam int IN_H = 224;
  localparam int IN_W = 224;
  localparam int C1_OUT_H = 112;
  localparam int C1_OUT_W = 112;
  localparam int MP_OUT_H = 56;
  localparam int MP_OUT_W = 56;
  localparam int L1_OUT_H = 56;
  localparam int L1_OUT_W = 56;
  localparam int L2_OUT_H = 28;
  localparam int L2_OUT_W = 28;
  localparam int L3_OUT_H = 14;
  localparam int L3_OUT_W = 14;
  localparam int L4_OUT_H = 7;
  localparam int L4_OUT_W = 7;

  logic signed [64*C1_OUT_H*C1_OUT_W*WIDTH-1:0] conv1_out;
  logic signed [64*C1_OUT_H*C1_OUT_W*WIDTH-1:0] bn1_out;
  logic signed [64*C1_OUT_H*C1_OUT_W*WIDTH-1:0] relu1_out;
  logic signed [64*MP_OUT_H*MP_OUT_W*WIDTH-1:0] maxpool_out;

  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_conv1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_bn1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_relu1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_conv2_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_bn2_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_add_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b0_out;

  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_conv1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_bn1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_relu1_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_conv2_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_bn2_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1b1_add_out;
  logic signed [64*L1_OUT_H*L1_OUT_W*WIDTH-1:0] l1_out;

  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_conv1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_bn1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_relu1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_conv2_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_bn2_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_ds_conv_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_ds_bn_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_add_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b0_out;

  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_conv1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_bn1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_relu1_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_conv2_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_bn2_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2b1_add_out;
  logic signed [128*L2_OUT_H*L2_OUT_W*WIDTH-1:0] l2_out;

  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_conv1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_bn1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_relu1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_conv2_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_bn2_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_ds_conv_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_ds_bn_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_add_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b0_out;

  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_conv1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_bn1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_relu1_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_conv2_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_bn2_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3b1_add_out;
  logic signed [256*L3_OUT_H*L3_OUT_W*WIDTH-1:0] l3_out;

  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_conv1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_bn1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_relu1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_conv2_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_bn2_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_ds_conv_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_ds_bn_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_add_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b0_out;

  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_conv1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_bn1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_relu1_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_conv2_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_bn2_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4b1_add_out;
  logic signed [512*L4_OUT_H*L4_OUT_W*WIDTH-1:0] l4_out;

  logic signed [512*1*1*WIDTH-1:0] gap_out;

  conv2d #(
    .IN_CH(3),
    .OUT_CH(64),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(7),
    .STRIDE(2),
    .PADDING(3),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(CONV1_WEIGHTS_FILE)
  ) conv1 (
    .in_vec(in_vec),
    .out_vec(conv1_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(C1_OUT_H),
    .IN_W(C1_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(BN1_SCALE_FILE),
    .BIAS_FILE(BN1_BIAS_FILE)
  ) bn1 (
    .in_vec(conv1_out),
    .out_vec(bn1_out)
  );

  relu #(
    .DIM(64*C1_OUT_H*C1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) relu1 (
    .in_vec(bn1_out),
    .out_vec(relu1_out)
  );

  maxpool2d #(
    .CH(64),
    .IN_H(C1_OUT_H),
    .IN_W(C1_OUT_W),
    .K(3),
    .STRIDE(2),
    .PADDING(1),
    .WIDTH(WIDTH),
    .precision(precision)
  ) maxpool (
    .in_vec(relu1_out),
    .out_vec(maxpool_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L1_0_CONV1_WEIGHTS_FILE)
  ) l1_0_conv1 (
    .in_vec(maxpool_out),
    .out_vec(l1b0_conv1_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L1_0_BN1_SCALE_FILE),
    .BIAS_FILE(L1_0_BN1_BIAS_FILE)
  ) l1_0_bn1 (
    .in_vec(l1b0_conv1_out),
    .out_vec(l1b0_bn1_out)
  );

  relu #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_0_relu1 (
    .in_vec(l1b0_bn1_out),
    .out_vec(l1b0_relu1_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L1_0_CONV2_WEIGHTS_FILE)
  ) l1_0_conv2 (
    .in_vec(l1b0_relu1_out),
    .out_vec(l1b0_conv2_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L1_0_BN2_SCALE_FILE),
    .BIAS_FILE(L1_0_BN2_BIAS_FILE)
  ) l1_0_bn2 (
    .in_vec(l1b0_conv2_out),
    .out_vec(l1b0_bn2_out)
  );

  add_vec #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_0_add (
    .a_vec(l1b0_bn2_out),
    .b_vec(maxpool_out),
    .out_vec(l1b0_add_out)
  );

  relu #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_0_relu2 (
    .in_vec(l1b0_add_out),
    .out_vec(l1b0_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L1_1_CONV1_WEIGHTS_FILE)
  ) l1_1_conv1 (
    .in_vec(l1b0_out),
    .out_vec(l1b1_conv1_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L1_1_BN1_SCALE_FILE),
    .BIAS_FILE(L1_1_BN1_BIAS_FILE)
  ) l1_1_bn1 (
    .in_vec(l1b1_conv1_out),
    .out_vec(l1b1_bn1_out)
  );

  relu #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_1_relu1 (
    .in_vec(l1b1_bn1_out),
    .out_vec(l1b1_relu1_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L1_1_CONV2_WEIGHTS_FILE)
  ) l1_1_conv2 (
    .in_vec(l1b1_relu1_out),
    .out_vec(l1b1_conv2_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L1_1_BN2_SCALE_FILE),
    .BIAS_FILE(L1_1_BN2_BIAS_FILE)
  ) l1_1_bn2 (
    .in_vec(l1b1_conv2_out),
    .out_vec(l1b1_bn2_out)
  );

  add_vec #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_1_add (
    .a_vec(l1b1_bn2_out),
    .b_vec(l1b0_out),
    .out_vec(l1b1_add_out)
  );

  relu #(
    .DIM(64*L1_OUT_H*L1_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l1_1_relu2 (
    .in_vec(l1b1_add_out),
    .out_vec(l1_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(128),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(3),
    .STRIDE(2),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_0_CONV1_WEIGHTS_FILE)
  ) l2_0_conv1 (
    .in_vec(l1_out),
    .out_vec(l2b0_conv1_out)
  );

  batchnorm2d #(
    .CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L2_0_BN1_SCALE_FILE),
    .BIAS_FILE(L2_0_BN1_BIAS_FILE)
  ) l2_0_bn1 (
    .in_vec(l2b0_conv1_out),
    .out_vec(l2b0_bn1_out)
  );

  relu #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_0_relu1 (
    .in_vec(l2b0_bn1_out),
    .out_vec(l2b0_relu1_out)
  );

  conv2d #(
    .IN_CH(128),
    .OUT_CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_0_CONV2_WEIGHTS_FILE)
  ) l2_0_conv2 (
    .in_vec(l2b0_relu1_out),
    .out_vec(l2b0_conv2_out)
  );

  batchnorm2d #(
    .CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L2_0_BN2_SCALE_FILE),
    .BIAS_FILE(L2_0_BN2_BIAS_FILE)
  ) l2_0_bn2 (
    .in_vec(l2b0_conv2_out),
    .out_vec(l2b0_bn2_out)
  );

  conv2d #(
    .IN_CH(64),
    .OUT_CH(128),
    .IN_H(L1_OUT_H),
    .IN_W(L1_OUT_W),
    .K(1),
    .STRIDE(2),
    .PADDING(0),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_0_DS_CONV_WEIGHTS_FILE)
  ) l2_0_ds_conv (
    .in_vec(l1_out),
    .out_vec(l2b0_ds_conv_out)
  );

  batchnorm2d #(
    .CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L2_0_DS_BN_SCALE_FILE),
    .BIAS_FILE(L2_0_DS_BN_BIAS_FILE)
  ) l2_0_ds_bn (
    .in_vec(l2b0_ds_conv_out),
    .out_vec(l2b0_ds_bn_out)
  );

  add_vec #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_0_add (
    .a_vec(l2b0_bn2_out),
    .b_vec(l2b0_ds_bn_out),
    .out_vec(l2b0_add_out)
  );

  relu #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_0_relu2 (
    .in_vec(l2b0_add_out),
    .out_vec(l2b0_out)
  );

  conv2d #(
    .IN_CH(128),
    .OUT_CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_1_CONV1_WEIGHTS_FILE)
  ) l2_1_conv1 (
    .in_vec(l2b0_out),
    .out_vec(l2b1_conv1_out)
  );

  batchnorm2d #(
    .CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L2_1_BN1_SCALE_FILE),
    .BIAS_FILE(L2_1_BN1_BIAS_FILE)
  ) l2_1_bn1 (
    .in_vec(l2b1_conv1_out),
    .out_vec(l2b1_bn1_out)
  );

  relu #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_1_relu1 (
    .in_vec(l2b1_bn1_out),
    .out_vec(l2b1_relu1_out)
  );

  conv2d #(
    .IN_CH(128),
    .OUT_CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L2_1_CONV2_WEIGHTS_FILE)
  ) l2_1_conv2 (
    .in_vec(l2b1_relu1_out),
    .out_vec(l2b1_conv2_out)
  );

  batchnorm2d #(
    .CH(128),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L2_1_BN2_SCALE_FILE),
    .BIAS_FILE(L2_1_BN2_BIAS_FILE)
  ) l2_1_bn2 (
    .in_vec(l2b1_conv2_out),
    .out_vec(l2b1_bn2_out)
  );

  add_vec #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_1_add (
    .a_vec(l2b1_bn2_out),
    .b_vec(l2b0_out),
    .out_vec(l2b1_add_out)
  );

  relu #(
    .DIM(128*L2_OUT_H*L2_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l2_1_relu2 (
    .in_vec(l2b1_add_out),
    .out_vec(l2_out)
  );

  conv2d #(
    .IN_CH(128),
    .OUT_CH(256),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .K(3),
    .STRIDE(2),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_0_CONV1_WEIGHTS_FILE)
  ) l3_0_conv1 (
    .in_vec(l2_out),
    .out_vec(l3b0_conv1_out)
  );

  batchnorm2d #(
    .CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L3_0_BN1_SCALE_FILE),
    .BIAS_FILE(L3_0_BN1_BIAS_FILE)
  ) l3_0_bn1 (
    .in_vec(l3b0_conv1_out),
    .out_vec(l3b0_bn1_out)
  );

  relu #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_0_relu1 (
    .in_vec(l3b0_bn1_out),
    .out_vec(l3b0_relu1_out)
  );

  conv2d #(
    .IN_CH(256),
    .OUT_CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_0_CONV2_WEIGHTS_FILE)
  ) l3_0_conv2 (
    .in_vec(l3b0_relu1_out),
    .out_vec(l3b0_conv2_out)
  );

  batchnorm2d #(
    .CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L3_0_BN2_SCALE_FILE),
    .BIAS_FILE(L3_0_BN2_BIAS_FILE)
  ) l3_0_bn2 (
    .in_vec(l3b0_conv2_out),
    .out_vec(l3b0_bn2_out)
  );

  conv2d #(
    .IN_CH(128),
    .OUT_CH(256),
    .IN_H(L2_OUT_H),
    .IN_W(L2_OUT_W),
    .K(1),
    .STRIDE(2),
    .PADDING(0),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_0_DS_CONV_WEIGHTS_FILE)
  ) l3_0_ds_conv (
    .in_vec(l2_out),
    .out_vec(l3b0_ds_conv_out)
  );

  batchnorm2d #(
    .CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L3_0_DS_BN_SCALE_FILE),
    .BIAS_FILE(L3_0_DS_BN_BIAS_FILE)
  ) l3_0_ds_bn (
    .in_vec(l3b0_ds_conv_out),
    .out_vec(l3b0_ds_bn_out)
  );

  add_vec #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_0_add (
    .a_vec(l3b0_bn2_out),
    .b_vec(l3b0_ds_bn_out),
    .out_vec(l3b0_add_out)
  );

  relu #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_0_relu2 (
    .in_vec(l3b0_add_out),
    .out_vec(l3b0_out)
  );

  conv2d #(
    .IN_CH(256),
    .OUT_CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_1_CONV1_WEIGHTS_FILE)
  ) l3_1_conv1 (
    .in_vec(l3b0_out),
    .out_vec(l3b1_conv1_out)
  );

  batchnorm2d #(
    .CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L3_1_BN1_SCALE_FILE),
    .BIAS_FILE(L3_1_BN1_BIAS_FILE)
  ) l3_1_bn1 (
    .in_vec(l3b1_conv1_out),
    .out_vec(l3b1_bn1_out)
  );

  relu #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_1_relu1 (
    .in_vec(l3b1_bn1_out),
    .out_vec(l3b1_relu1_out)
  );

  conv2d #(
    .IN_CH(256),
    .OUT_CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L3_1_CONV2_WEIGHTS_FILE)
  ) l3_1_conv2 (
    .in_vec(l3b1_relu1_out),
    .out_vec(l3b1_conv2_out)
  );

  batchnorm2d #(
    .CH(256),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L3_1_BN2_SCALE_FILE),
    .BIAS_FILE(L3_1_BN2_BIAS_FILE)
  ) l3_1_bn2 (
    .in_vec(l3b1_conv2_out),
    .out_vec(l3b1_bn2_out)
  );

  add_vec #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_1_add (
    .a_vec(l3b1_bn2_out),
    .b_vec(l3b0_out),
    .out_vec(l3b1_add_out)
  );

  relu #(
    .DIM(256*L3_OUT_H*L3_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l3_1_relu2 (
    .in_vec(l3b1_add_out),
    .out_vec(l3_out)
  );

  conv2d #(
    .IN_CH(256),
    .OUT_CH(512),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .K(3),
    .STRIDE(2),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L4_0_CONV1_WEIGHTS_FILE)
  ) l4_0_conv1 (
    .in_vec(l3_out),
    .out_vec(l4b0_conv1_out)
  );

  batchnorm2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L4_0_BN1_SCALE_FILE),
    .BIAS_FILE(L4_0_BN1_BIAS_FILE)
  ) l4_0_bn1 (
    .in_vec(l4b0_conv1_out),
    .out_vec(l4b0_bn1_out)
  );

  relu #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_0_relu1 (
    .in_vec(l4b0_bn1_out),
    .out_vec(l4b0_relu1_out)
  );

  conv2d #(
    .IN_CH(512),
    .OUT_CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L4_0_CONV2_WEIGHTS_FILE)
  ) l4_0_conv2 (
    .in_vec(l4b0_relu1_out),
    .out_vec(l4b0_conv2_out)
  );

  batchnorm2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L4_0_BN2_SCALE_FILE),
    .BIAS_FILE(L4_0_BN2_BIAS_FILE)
  ) l4_0_bn2 (
    .in_vec(l4b0_conv2_out),
    .out_vec(l4b0_bn2_out)
  );

  conv2d #(
    .IN_CH(256),
    .OUT_CH(512),
    .IN_H(L3_OUT_H),
    .IN_W(L3_OUT_W),
    .K(1),
    .STRIDE(2),
    .PADDING(0),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L4_0_DS_CONV_WEIGHTS_FILE)
  ) l4_0_ds_conv (
    .in_vec(l3_out),
    .out_vec(l4b0_ds_conv_out)
  );

  batchnorm2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L4_0_DS_BN_SCALE_FILE),
    .BIAS_FILE(L4_0_DS_BN_BIAS_FILE)
  ) l4_0_ds_bn (
    .in_vec(l4b0_ds_conv_out),
    .out_vec(l4b0_ds_bn_out)
  );

  add_vec #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_0_add (
    .a_vec(l4b0_bn2_out),
    .b_vec(l4b0_ds_bn_out),
    .out_vec(l4b0_add_out)
  );

  relu #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_0_relu2 (
    .in_vec(l4b0_add_out),
    .out_vec(l4b0_out)
  );

  conv2d #(
    .IN_CH(512),
    .OUT_CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L4_1_CONV1_WEIGHTS_FILE)
  ) l4_1_conv1 (
    .in_vec(l4b0_out),
    .out_vec(l4b1_conv1_out)
  );

  batchnorm2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L4_1_BN1_SCALE_FILE),
    .BIAS_FILE(L4_1_BN1_BIAS_FILE)
  ) l4_1_bn1 (
    .in_vec(l4b1_conv1_out),
    .out_vec(l4b1_bn1_out)
  );

  relu #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_1_relu1 (
    .in_vec(l4b1_bn1_out),
    .out_vec(l4b1_relu1_out)
  );

  conv2d #(
    .IN_CH(512),
    .OUT_CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(L4_1_CONV2_WEIGHTS_FILE)
  ) l4_1_conv2 (
    .in_vec(l4b1_relu1_out),
    .out_vec(l4b1_conv2_out)
  );

  batchnorm2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .SCALE_FILE(L4_1_BN2_SCALE_FILE),
    .BIAS_FILE(L4_1_BN2_BIAS_FILE)
  ) l4_1_bn2 (
    .in_vec(l4b1_conv2_out),
    .out_vec(l4b1_bn2_out)
  );

  add_vec #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_1_add (
    .a_vec(l4b1_bn2_out),
    .b_vec(l4b0_out),
    .out_vec(l4b1_add_out)
  );

  relu #(
    .DIM(512*L4_OUT_H*L4_OUT_W),
    .WIDTH(WIDTH),
    .precision(precision)
  ) l4_1_relu2 (
    .in_vec(l4b1_add_out),
    .out_vec(l4_out)
  );

  avgpool2d #(
    .CH(512),
    .IN_H(L4_OUT_H),
    .IN_W(L4_OUT_W),
    .K(7),
    .STRIDE(1),
    .WIDTH(WIDTH),
    .precision(precision)
  ) gap (
    .in_vec(l4_out),
    .out_vec(gap_out)
  );

  linear #(
    .IN_DIM(512),
    .OUT_DIM(1000),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(FC_WEIGHTS_FILE),
    .BIAS_FILE(FC_BIAS_FILE)
  ) fc (
    .in_vec(gap_out),
    .out_vec(out_vec)
  );
endmodule
