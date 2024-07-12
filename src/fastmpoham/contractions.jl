@tightloop_planar tight_ac_contractor out[-1 -2;-3] += left[-1 -2 4;1 2]*x[1 2;3]*right[3 4;-3]

@tightloop_planar tight_c_contractor out[-1;-2] += left[-1 2;1]*x[1;3]*right[3 2;-2]

@tightloop_planar fast_reg_bond v[-1; -2] -= lvec[1; 2] * v[2; 1] * rvec[-1; -2]


@tightloop_planar fast_reg_mps v[-1 -2; -3] -= lvec[1; 2] * v[2 -2; 1] * rvec[-1; -3]

@tightloop_planar fast_left_bond y[-1;-2] := v[1;2]*a[2 3;-2]*b[-1;1 3]
@tightloop_planar fast_right_bond y[-1;-2] := v[1;2]*a[-1 3;1]*b[2;-2 3]

@tightloop_planar fast_left_mps y[-1 -2;-3] :=  v[1 2; 4] * a[4 5; -3] * braid[2 3; 5 -2] * b[-1;1 3]
@tightloop_planar fast_right_mps y[-1 -2;-3] := a[-1 2; 1] * braid[-2 4; 2 3] * b[5;-3 4] * v[1 3; 5]
