module tanh #(
  parameter int DIM = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8"
) (
  input  logic signed [DIM*WIDTH-1:0] in_vec,
  output logic signed [DIM*WIDTH-1:0] out_vec
);
  integer i;
  logic signed [WIDTH-1:0] in_elem;
  logic signed [WIDTH-1:0] out_elem;
  real scaled;
  real x;
  real y;
  integer q;

  localparam real SCALE = 1.0 / (1 << FRAC);
  localparam string PRECISION = precision;

  always_comb begin
    out_vec = '0;
    for (i = 0; i < DIM; i = i + 1) begin
      in_elem = in_vec[i*WIDTH +: WIDTH];
      x = $itor(in_elem) * SCALE;
      y = $tanh(x);
      scaled = y * (1 << FRAC);
      if (scaled >= 0.0) begin
        q = $rtoi(scaled + 0.5);
      end else begin
        q = $rtoi(scaled - 0.5);
      end
      out_elem = q[WIDTH-1:0];
      out_vec[i*WIDTH +: WIDTH] = out_elem;
    end
  end
endmodule
