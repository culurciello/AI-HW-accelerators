module tanh #(
  parameter int DIM = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter int LUT_SIZE = 1024,
  parameter int X_MIN_Q = -(4 << FRAC),
  parameter int X_MAX_Q = (4 << FRAC)
) (
  input  logic signed [DIM*WIDTH-1:0] in_vec,
  output logic signed [DIM*WIDTH-1:0] out_vec
);
  localparam real SCALE = 1.0 / (1 << FRAC);
  localparam int RANGE_Q = X_MAX_Q - X_MIN_Q;

  logic signed [WIDTH-1:0] lut [0:LUT_SIZE-1];

  integer i;
  integer j;
  integer signed_val;
  real x;
  real y;
  real scaled;
  integer q;
  integer signed x_in;
  integer signed x_clamped;
  integer signed offset;
  integer idx;

  localparam string PRECISION = precision;

  initial begin
    for (i = 0; i < LUT_SIZE; i = i + 1) begin
      signed_val = X_MIN_Q + (i * RANGE_Q) / (LUT_SIZE - 1);
      x = $itor(signed_val) * SCALE;
      y = $tanh(x);
      scaled = y * (1 << FRAC);
      if (scaled >= 0.0) begin
        q = $rtoi(scaled + 0.5);
      end else begin
        q = $rtoi(scaled - 0.5);
      end
      lut[i] = q[WIDTH-1:0];
    end
  end

  always_comb begin
    out_vec = '0;
    for (j = 0; j < DIM; j = j + 1) begin
      x_in = $signed(in_vec[j*WIDTH +: WIDTH]);
      if (x_in < X_MIN_Q) begin
        x_clamped = X_MIN_Q;
      end else if (x_in > X_MAX_Q) begin
        x_clamped = X_MAX_Q;
      end else begin
        x_clamped = x_in;
      end
      offset = x_clamped - X_MIN_Q;
      idx = (offset * (LUT_SIZE - 1)) / RANGE_Q;
      if (idx < 0) begin
        idx = 0;
      end else if (idx > (LUT_SIZE - 1)) begin
        idx = LUT_SIZE - 1;
      end
      out_vec[j*WIDTH +: WIDTH] = lut[idx];
    end
  end
endmodule
