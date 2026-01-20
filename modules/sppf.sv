module sppf #(
  parameter int IN_CH = 1,
  parameter int OUT_CH = 1,
  parameter int MID_CH = (IN_CH/2),
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string CV1_WEIGHTS_FILE = "",
  parameter string CV1_BN_SCALE_FILE = "",
  parameter string CV1_BN_BIAS_FILE = "",
  parameter string CV2_WEIGHTS_FILE = "",
  parameter string CV2_BN_SCALE_FILE = "",
  parameter string CV2_BN_BIAS_FILE = ""
) (
  input  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] out_vec
);
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] cv1_out;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] y1;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] y2;
  logic signed [MID_CH*IN_H*IN_W*WIDTH-1:0] y3;
  logic signed [4*MID_CH*IN_H*IN_W*WIDTH-1:0] cat_vec;

  yolo_conv #(
    .IN_CH(IN_CH),
    .OUT_CH(MID_CH),
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

  maxpool2d #(
    .CH(MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(5),
    .STRIDE(1),
    .PADDING(2),
    .WIDTH(WIDTH),
    .precision(precision)
  ) mp1 (
    .in_vec(cv1_out),
    .out_vec(y1)
  );

  maxpool2d #(
    .CH(MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(5),
    .STRIDE(1),
    .PADDING(2),
    .WIDTH(WIDTH),
    .precision(precision)
  ) mp2 (
    .in_vec(y1),
    .out_vec(y2)
  );

  maxpool2d #(
    .CH(MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(5),
    .STRIDE(1),
    .PADDING(2),
    .WIDTH(WIDTH),
    .precision(precision)
  ) mp3 (
    .in_vec(y2),
    .out_vec(y3)
  );

  always_comb begin
    cat_vec = '0;
    cat_vec[0 +: MID_CH*IN_H*IN_W*WIDTH] = cv1_out;
    cat_vec[MID_CH*IN_H*IN_W*WIDTH +: MID_CH*IN_H*IN_W*WIDTH] = y1;
    cat_vec[2*MID_CH*IN_H*IN_W*WIDTH +: MID_CH*IN_H*IN_W*WIDTH] = y2;
    cat_vec[3*MID_CH*IN_H*IN_W*WIDTH +: MID_CH*IN_H*IN_W*WIDTH] = y3;
  end

  yolo_conv #(
    .IN_CH(4*MID_CH),
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
