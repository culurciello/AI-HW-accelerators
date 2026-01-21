module maxpool2d #(
  parameter int CH = 1,
  parameter int IN_H = 2,
  parameter int IN_W = 2,
  parameter int K = 2,
  parameter int STRIDE = 2,
  parameter int PADDING = 0,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [CH*((IN_H+2*PADDING-K)/STRIDE+1)*((IN_W+2*PADDING-K)/STRIDE+1)*WIDTH-1:0] out_vec
);
  localparam int OUT_H = (IN_H + 2*PADDING - K) / STRIDE + 1;
  localparam int OUT_W = (IN_W + 2*PADDING - K) / STRIDE + 1;
  localparam logic signed [WIDTH-1:0] PAD_VAL = {1'b1, {(WIDTH-1){1'b0}}};

  function automatic logic signed [WIDTH-1:0] get_in(
    input int ch,
    input int h,
    input int w
  );
    int idx;
    begin
      if (h < 0 || h >= IN_H || w < 0 || w >= IN_W) begin
        get_in = PAD_VAL;
      end else begin
        idx = ((ch * IN_H + h) * IN_W + w);
        get_in = in_vec[idx*WIDTH +: WIDTH];
      end
    end
  endfunction

  integer c;
  integer oh;
  integer ow;
  integer kh;
  integer kw;
  logic signed [WIDTH-1:0] max_val;
  logic signed [WIDTH-1:0] in_val;

  localparam string PRECISION = precision;

  always_comb begin
    for (c = 0; c < CH; c = c + 1) begin
      for (oh = 0; oh < OUT_H; oh = oh + 1) begin
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin
          max_val = PAD_VAL;
          for (kh = 0; kh < K; kh = kh + 1) begin
            for (kw = 0; kw < K; kw = kw + 1) begin
              in_val = get_in(c, oh*STRIDE + kh - PADDING, ow*STRIDE + kw - PADDING);
              if (in_val > max_val) begin
                max_val = in_val;
              end
            end
          end
          out_vec[((c*OUT_H + oh)*OUT_W + ow)*WIDTH +: WIDTH] = max_val;
        end
      end
    end
  end
endmodule
