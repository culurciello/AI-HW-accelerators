module concat2d #(
  parameter int A_CH = 1,
  parameter int B_CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [A_CH*IN_H*IN_W*WIDTH-1:0] a_vec,
  input  logic signed [B_CH*IN_H*IN_W*WIDTH-1:0] b_vec,
  output logic signed [(A_CH+B_CH)*IN_H*IN_W*WIDTH-1:0] out_vec
);
  localparam int A_SIZE = A_CH*IN_H*IN_W*WIDTH;
  localparam int B_SIZE = B_CH*IN_H*IN_W*WIDTH;

  localparam string PRECISION = precision;

  always_comb begin
    out_vec = '0;
    out_vec[0 +: A_SIZE] = a_vec;
    out_vec[A_SIZE +: B_SIZE] = b_vec;
  end
endmodule
