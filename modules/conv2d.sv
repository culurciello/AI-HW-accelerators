module conv2d #(
  parameter int IN_CH = 1,
  parameter int OUT_CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int K = 3,
  parameter int WIDTH = 16,
  parameter int FRAC = 8,
  parameter string precision = "Q8.8",
  parameter string WEIGHTS_FILE = "",
  parameter string BIAS_FILE = ""
) (
  input  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [OUT_CH*(IN_H-K+1)*(IN_W-K+1)*WIDTH-1:0] out_vec
);
  localparam int OUT_H = IN_H - K + 1;
  localparam int OUT_W = IN_W - K + 1;
  localparam int ACC_WIDTH = WIDTH*2 + $clog2(IN_CH*K*K);

  logic signed [WIDTH-1:0] weights [0:OUT_CH*IN_CH*K*K-1];
  logic signed [WIDTH-1:0] bias [0:OUT_CH-1];

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

  function automatic logic signed [WIDTH-1:0] get_weight(
    input int out_ch,
    input int in_ch,
    input int kh,
    input int kw
  );
    int idx;
    begin
      idx = (((out_ch * IN_CH + in_ch) * K + kh) * K + kw);
      get_weight = weights[idx];
    end
  endfunction

  function automatic logic signed [WIDTH-1:0] get_bias(
    input int out_ch
  );
    get_bias = bias[out_ch];
  endfunction

  integer oc;
  integer ic;
  integer oh;
  integer ow;
  integer kh;
  integer kw;
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
  localparam string BIAS_SRC = BIAS_FILE;

  initial begin
    for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
      bias[oc] = '0;
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
    for (oc = 0; oc < OUT_CH; oc = oc + 1) begin
      for (oh = 0; oh < OUT_H; oh = oh + 1) begin
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin
          acc = '0;
          for (ic = 0; ic < IN_CH; ic = ic + 1) begin
            for (kh = 0; kh < K; kh = kh + 1) begin
              for (kw = 0; kw < K; kw = kw + 1) begin
                prod = get_in(ic, oh + kh, ow + kw) * get_weight(oc, ic, kh, kw);
                prod_ext = {{(ACC_WIDTH-WIDTH*2){prod[WIDTH*2-1]}}, prod};
                acc = acc + prod_ext;
              end
            end
          end
          bias_ext = {{(ACC_WIDTH-WIDTH){get_bias(oc)[WIDTH-1]}}, get_bias(oc)};
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
          out_vec[((oc*OUT_H + oh)*OUT_W + ow)*WIDTH +: WIDTH] = out_elem;
        end
      end
    end
  end
endmodule
