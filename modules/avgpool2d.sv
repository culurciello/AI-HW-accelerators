module avgpool2d #(
  parameter int CH = 1,
  parameter int IN_H = 2,
  parameter int IN_W = 2,
  parameter int K = 2,
  parameter int STRIDE = 2,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [CH*((IN_H-K)/STRIDE+1)*((IN_W-K)/STRIDE+1)*WIDTH-1:0] out_vec
);
  localparam int OUT_H = (IN_H - K) / STRIDE + 1;
  localparam int OUT_W = (IN_W - K) / STRIDE + 1;
  localparam int ACC_WIDTH = WIDTH + $clog2(K*K) + 1;
  localparam int EXT_WIDTH = ACC_WIDTH + 1;
  localparam int DENOM = K*K;

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

  function automatic logic get_in_sign(
    input int ch,
    input int h,
    input int w
  );
    int idx;
    begin
      idx = ((ch * IN_H + h) * IN_W + w);
      get_in_sign = in_vec[idx*WIDTH + WIDTH-1];
    end
  endfunction

  integer c;
  integer oh;
  integer ow;
  integer kh;
  integer kw;
  logic signed [ACC_WIDTH-1:0] acc;
  logic signed [ACC_WIDTH-1:0] abs_acc;
  logic signed [ACC_WIDTH-1:0] rounded_mag;
  logic signed [EXT_WIDTH-1:0] acc_ext;
  logic signed [EXT_WIDTH-1:0] abs_ext;
  logic signed [WIDTH-1:0] out_elem;

  localparam string PRECISION = precision;

  always_comb begin
    /* verilator lint_off WIDTHCONCAT */
    out_vec = '0;
    /* verilator lint_on WIDTHCONCAT */
    for (c = 0; c < CH; c = c + 1) begin
      for (oh = 0; oh < OUT_H; oh = oh + 1) begin
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin
          acc = '0;
          for (kh = 0; kh < K; kh = kh + 1) begin
            for (kw = 0; kw < K; kw = kw + 1) begin
              acc = acc + {{(ACC_WIDTH-WIDTH){get_in_sign(c, oh*STRIDE + kh, ow*STRIDE + kw)}}, get_in(c, oh*STRIDE + kh, ow*STRIDE + kw)};
            end
          end
          if (acc[ACC_WIDTH-1]) begin
            abs_acc = -acc;
            abs_ext = {{(EXT_WIDTH-ACC_WIDTH){abs_acc[ACC_WIDTH-1]}}, abs_acc};
            rounded_mag = (abs_ext + (DENOM/2)) / DENOM;
            out_elem = -rounded_mag[WIDTH-1:0];
          end else begin
            acc_ext = {{(EXT_WIDTH-ACC_WIDTH){acc[ACC_WIDTH-1]}}, acc};
            rounded_mag = (acc_ext + (DENOM/2)) / DENOM;
            out_elem = rounded_mag[WIDTH-1:0];
          end
          out_vec[((c*OUT_H + oh)*OUT_W + ow)*WIDTH +: WIDTH] = out_elem;
        end
      end
    end
  end
endmodule
