module bottleneck #(
  parameter int IN_CH = 1,
  parameter int OUT_CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter bit SHORTCUT = 1'b1,
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
  logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] cv1_out;
  logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] cv2_out;

  yolo_conv #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
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

  yolo_conv #(
    .IN_CH(OUT_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .precision(precision),
    .WEIGHTS_FILE(CV2_WEIGHTS_FILE),
    .BN_SCALE_FILE(CV2_BN_SCALE_FILE),
    .BN_BIAS_FILE(CV2_BN_BIAS_FILE)
  ) cv2 (
    .in_vec(cv1_out),
    .out_vec(cv2_out)
  );

  generate
    if (SHORTCUT && (IN_CH == OUT_CH)) begin : gen_shortcut
      add_vec #(
        .DIM(OUT_CH*IN_H*IN_W),
        .WIDTH(WIDTH),
        .precision(precision)
      ) add (
        .a_vec(in_vec),
        .b_vec(cv2_out),
        .out_vec(out_vec)
      );
    end else begin : gen_no_shortcut
      assign out_vec = cv2_out;
    end
  endgenerate
endmodule
