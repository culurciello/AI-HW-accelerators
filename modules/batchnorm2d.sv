module batchnorm2d #(
  parameter int CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string SCALE_FILE = "",
  parameter string BIAS_FILE = ""
) (
  input  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [CH*IN_H*IN_W*WIDTH-1:0] out_vec
);
  localparam int ACC_WIDTH = WIDTH*2 + 1;

  logic signed [WIDTH-1:0] scale [0:CH-1];
  logic signed [WIDTH-1:0] bias [0:CH-1];

  function automatic logic signed [WIDTH-1:0] get_in(
    input int ch,
    input int h,
    input int w
  );
    int idx;
    begin
      idx = ((ch * IN_H + h) * IN_W + w);
      get_in = in_vec[idx*WIDTH +: WIDTH];
    end
  endfunction

  function automatic logic signed [WIDTH-1:0] get_scale(
    input int ch
  );
    get_scale = scale[ch];
  endfunction

  function automatic logic signed [WIDTH-1:0] get_bias(
    input int ch
  );
    get_bias = bias[ch];
  endfunction

  function automatic logic get_bias_sign(
    input int ch
  );
    get_bias_sign = bias[ch][WIDTH-1];
  endfunction

  integer c;
  integer h;
  integer w;
  logic signed [ACC_WIDTH-1:0] acc;
  logic signed [ACC_WIDTH-1:0] acc_round;
  logic signed [WIDTH*2-1:0] prod;
  logic signed [ACC_WIDTH-1:0] prod_ext;
  logic signed [ACC_WIDTH-1:0] rounding;
  logic signed [ACC_WIDTH-1:0] shifted;
  logic signed [ACC_WIDTH-1:0] abs_acc;
  logic signed [ACC_WIDTH-1:0] rounded_mag;
  logic signed [ACC_WIDTH-1:0] bias_ext;
  logic signed [WIDTH-1:0] out_elem;

  localparam string PRECISION = precision;
  localparam string SCALE_SRC = SCALE_FILE;
  localparam string BIAS_SRC = BIAS_FILE;

  initial begin
    for (c = 0; c < CH; c = c + 1) begin
      scale[c] = '0;
      bias[c] = '0;
    end
    if (SCALE_FILE != "") begin
      $readmemh(SCALE_FILE, scale);
    end
    if (BIAS_FILE != "") begin
      $readmemh(BIAS_FILE, bias);
    end
  end

  always_comb begin
    out_vec = '0;
    rounding = ({{(ACC_WIDTH-1){1'b0}}, 1'b1} <<< (FRAC-1));
    for (c = 0; c < CH; c = c + 1) begin
      for (h = 0; h < IN_H; h = h + 1) begin
        for (w = 0; w < IN_W; w = w + 1) begin
          prod = get_in(c, h, w) * get_scale(c);
          prod_ext = {{(ACC_WIDTH-WIDTH*2){prod[WIDTH*2-1]}}, prod};
          bias_ext = {{(ACC_WIDTH-WIDTH){get_bias_sign(c)}}, get_bias(c)};
          acc = prod_ext + (bias_ext <<< FRAC);
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
          out_vec[((c*IN_H + h)*IN_W + w)*WIDTH +: WIDTH] = out_elem;
        end
      end
    end
  end
endmodule
