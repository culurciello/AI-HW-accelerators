module relu #(
  parameter int DIM = 1,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [DIM*WIDTH-1:0] in_vec,
  output logic signed [DIM*WIDTH-1:0] out_vec
);
  integer i;
  logic signed [WIDTH-1:0] in_elem;
  localparam string PRECISION = precision;

  always_comb begin
    out_vec = '0;
    for (i = 0; i < DIM; i = i + 1) begin
      in_elem = in_vec[i*WIDTH +: WIDTH];
      if (in_elem[WIDTH-1]) begin
        out_vec[i*WIDTH +: WIDTH] = '0;
      end else begin
        out_vec[i*WIDTH +: WIDTH] = in_elem;
      end
    end
  end
endmodule
