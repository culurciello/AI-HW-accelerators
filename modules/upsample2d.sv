module upsample2d #(
  parameter int CH = 1,
  parameter int IN_H = 1,
  parameter int IN_W = 1,
  parameter int SCALE = 2,
  parameter int WIDTH = 16,
  parameter string precision = "Q8.8"
) (
  input  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec,
  output logic signed [CH*(IN_H*SCALE)*(IN_W*SCALE)*WIDTH-1:0] out_vec
);
  localparam int OUT_H = IN_H * SCALE;
  localparam int OUT_W = IN_W * SCALE;

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

  integer c;
  integer oh;
  integer ow;
  integer ih;
  integer iw;

  localparam string PRECISION = precision;

  always_comb begin
    for (c = 0; c < CH; c = c + 1) begin
      for (oh = 0; oh < OUT_H; oh = oh + 1) begin
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin
          ih = oh / SCALE;
          iw = ow / SCALE;
          out_vec[((c*OUT_H + oh)*OUT_W + ow)*WIDTH +: WIDTH] = get_in(c, ih, iw);
        end
      end
    end
  end
endmodule
