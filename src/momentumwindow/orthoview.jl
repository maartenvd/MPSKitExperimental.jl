FiniteMultiline = Multiline{<:FiniteMPS};

function Base.getproperty(st::FiniteMultiline,s::Symbol)
    if s == :AL
        MPSKit.ALView(st);
    elseif s == :AR
        MPSKit.ARView(st);
    elseif s == :C
        MPSKit.CView(st);
    elseif s == :AC
        MPSKit.ACView(st);
    else
        getfield(st,s);
    end
end
