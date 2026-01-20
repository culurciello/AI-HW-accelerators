module silu #(
  parameter int DIM = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter int LUT_SIZE = 1024,
  parameter int X_MIN_Q = -(8 << FRAC),
  parameter int X_MAX_Q = (8 << FRAC)
) (
  input  logic signed [DIM*WIDTH-1:0] in_vec,
  output logic signed [DIM*WIDTH-1:0] out_vec
);
  localparam int RANGE_Q = X_MAX_Q - X_MIN_Q;
  localparam int ACC_WIDTH = WIDTH*2 + 1;

  logic signed [WIDTH-1:0] sig_lut [0:LUT_SIZE-1];

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

  logic signed [WIDTH-1:0] sig_q;
  logic signed [WIDTH*2-1:0] prod;
  logic signed [ACC_WIDTH-1:0] prod_ext;
  logic signed [ACC_WIDTH-1:0] acc_round;
  logic signed [ACC_WIDTH-1:0] abs_acc;
  logic signed [ACC_WIDTH-1:0] rounded_mag;
  logic signed [ACC_WIDTH-1:0] rounding;
  logic signed [WIDTH-1:0] out_elem;

  localparam string PRECISION = precision;

  initial begin
    for (i = 0; i < LUT_SIZE; i = i + 1) begin
      signed_val = X_MIN_Q + (i * RANGE_Q) / (LUT_SIZE - 1);
      x = $itor(signed_val) / (1 << FRAC);
      y = 1.0 / (1.0 + $exp(-x));
      scaled = y * (1 << FRAC);
      if (scaled >= 0.0) begin
        q = $rtoi(scaled + 0.5);
      end else begin
        q = $rtoi(scaled - 0.5);
      end
      sig_lut[i] = q[WIDTH-1:0];
    end
  end

  always_comb begin
    out_vec = '0;
    rounding = ({{(ACC_WIDTH-1){1'b0}}, 1'b1} <<< (FRAC-1));
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
      sig_q = sig_lut[idx];
      prod = $signed(in_vec[j*WIDTH +: WIDTH]) * $signed(sig_q);
      prod_ext = {{(ACC_WIDTH-WIDTH*2){prod[WIDTH*2-1]}}, prod};
      if (prod_ext[ACC_WIDTH-1]) begin
        abs_acc = -prod_ext;
        acc_round = abs_acc + rounding;
        rounded_mag = acc_round >>> FRAC;
        out_elem = -rounded_mag[WIDTH-1:0];
      end else begin
        acc_round = prod_ext + rounding;
        rounded_mag = acc_round >>> FRAC;
        out_elem = rounded_mag[WIDTH-1:0];
      end
      out_vec[j*WIDTH +: WIDTH] = out_elem;
    end
  end
endmodule
