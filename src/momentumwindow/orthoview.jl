FiniteMultiline = Multiline{<:FiniteMPS};

function Base.getproperty(st::FiniteMultiline,s::Symbol)
    if s == :AL
        MPSKit.ALView(st);
    elseif s == :AR
        MPSKit.ARView(st);
    elseif s == :CR
        MPSKit.CRView(st);
    elseif s == :AC
        MPSKit.ACView(st);
    else
        getfield(st,s);
    end
end
