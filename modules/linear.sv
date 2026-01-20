module linear #(
  parameter int IN_DIM = 1,
  parameter int OUT_DIM = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string WEIGHTS_FILE = "",
  parameter string BIAS_FILE = ""
) (
  input  logic signed [IN_DIM*WIDTH-1:0] in_vec,
  output logic signed [OUT_DIM*WIDTH-1:0] out_vec
);
  localparam int ACC_WIDTH = WIDTH*2 + $clog2(IN_DIM);

  logic signed [WIDTH-1:0] weights [0:OUT_DIM*IN_DIM-1];
  logic signed [WIDTH-1:0] bias [0:OUT_DIM-1];

  function automatic logic signed [WIDTH-1:0] get_in(
    input logic signed [IN_DIM*WIDTH-1:0] vec,
    input int idx
  );
    get_in = vec[idx*WIDTH +: WIDTH];
  endfunction

  function automatic logic signed [WIDTH-1:0] get_weight(
    input int out_idx,
    input int in_idx
  );
    get_weight = weights[out_idx*IN_DIM + in_idx];
  endfunction

  function automatic logic signed [WIDTH-1:0] get_bias(
    input int out_idx
  );
    get_bias = bias[out_idx];
  endfunction

  function automatic logic get_bias_sign(
    input int out_idx
  );
    get_bias_sign = bias[out_idx][WIDTH-1];
  endfunction

  integer o;
  integer i;
  logic signed [ACC_WIDTH-1:0] acc;
  logic signed [ACC_WIDTH-1:0] acc_round;
  logic signed [WIDTH-1:0] out_elem;
  logic signed [WIDTH*2-1:0] prod;
  logic signed [ACC_WIDTH-1:0] prod_ext;
  logic signed [ACC_WIDTH-1:0] rounding;
  logic signed [ACC_WIDTH-1:0] shifted;
  logic signed [ACC_WIDTH-1:0] abs_acc;
  logic signed [ACC_WIDTH-1:0] rounded_mag;
  logic signed [ACC_WIDTH-1:0] bias_ext;

  localparam string PRECISION = precision;
  localparam string BIAS_SRC = BIAS_FILE;

  initial begin
    for (o = 0; o < OUT_DIM; o = o + 1) begin
      bias[o] = '0;
    end
    if (WEIGHTS_FILE != "") begin
      $readmemh(WEIGHTS_FILE, weights);
    end
    if (BIAS_FILE != "") begin
      $readmemh(BIAS_FILE, bias);
    end
  end

  always_comb begin
    out_vec = '0;
    rounding = ({{(ACC_WIDTH-1){1'b0}}, 1'b1} <<< (FRAC-1));
    for (o = 0; o < OUT_DIM; o = o + 1) begin
      acc = '0;
      for (i = 0; i < IN_DIM; i = i + 1) begin
        prod = get_in(in_vec, i) * get_weight(o, i);
        prod_ext = {{(ACC_WIDTH-WIDTH*2){prod[WIDTH*2-1]}}, prod};
        acc = acc + prod_ext;
      end
      bias_ext = {{(ACC_WIDTH-WIDTH){get_bias_sign(o)}}, get_bias(o)};
      acc = acc + (bias_ext <<< FRAC);
      if (acc[ACC_WIDTH-1]) begin
        abs_acc = -acc;
        acc_round = abs_acc + rounding;
        shifted = acc_round >>> FRAC;
        rounded_mag = shifted;
        out_elem = -rounded_mag[WIDTH-1:0];
      end else begin
        acc_round = acc + rounding;
        shifted = acc_round >>> FRAC;
        out_elem = shifted[WIDTH-1:0];
      end
      out_vec[o*WIDTH +: WIDTH] = out_elem;
    end
  end
endmodule
