module add_vec #(
  parameter int DIM = 1,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [DIM*WIDTH-1:0] a_vec,
  input  logic signed [DIM*WIDTH-1:0] b_vec,
  output logic signed [DIM*WIDTH-1:0] out_vec
);
  integer i;
  logic signed [WIDTH-1:0] a_elem;
  logic signed [WIDTH-1:0] b_elem;
  logic signed [WIDTH-1:0] sum_elem;

  localparam string PRECISION = precision;

  always_comb begin
    out_vec = '0;
    for (i = 0; i < DIM; i = i + 1) begin
      a_elem = a_vec[i*WIDTH +: WIDTH];
      b_elem = b_vec[i*WIDTH +: WIDTH];
      sum_elem = a_elem + b_elem;
      out_vec[i*WIDTH +: WIDTH] = sum_elem;
    end
  end
endmodule
