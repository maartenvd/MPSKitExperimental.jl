using Base.Iterators;

function parse_fcidump(file)
    lines = eachline(file);
    
    # build header:
    (header,_) = iterate(lines);
    while !contains(header,"&END")
        (line,_) = iterate(lines);
        isempty(lines) && throw(ErrorException("exhausted file searching for header terminator"));
        
        line = replace(string(line),"/"=>"&END")
        header = string(header,line);
    end
    
    NORB_r = r"[\ ,]NORB\ *=\ *[0-9]+[, ]";
    pl = match(NORB_r,header).match;
    NORB = parse(Int,string(filter(x->isnumeric(x),collect(pl))...));
    
    NELEC_r = r"[\ ,]NELEC\ *=\ *[0-9]+[, ]";
    pl = match(NELEC_r,header).match;
    NELEC = parse(Int,string(filter(x->isnumeric(x),collect(pl))...));

    MS2_r = r"[\ ,]MS2\ *=\ *[0-9]+[, ]";
    pl = match(MS2_r,header).match;
    MS2 = parse(Int,string(filter(x->isnumeric(x),collect(pl))[2:end]...));

    # header is parsed, let's turn to the ERI/T/E0
    ERI = fill(0.0+0im,NORB,NORB,NORB,NORB);
    K = fill(0.0+0im,NORB,NORB);
    E0 = 0.0+0im;

    for l in lines
        nl = replace(l,"  "=>" ");
        while nl!=l
            l = nl
            nl = replace(l,"  "=>" ");
        end
        while startswith(l," ")
            l = chop(l,head=1,tail=0)
        end
        while endswith(l," ")
            l = chop(l,head=0,tail=1)
        end

        spl = split(l," ");
        @assert length(spl) == 5;
        val = parse(ComplexF64,spl[1]);

        (i,a,j,b) = tuple(parse.(Int,spl[2:end])...)
        
        if all(iszero.([i,a,j,b]))
            E0 = val
        elseif a==0 && j == 0 && b == 0
            @show "I don't know"
            @show i,val
        elseif j==0 && b == 0
            K[i,a] = val
            K[a,i] = val'
        else
            @assert abs(real(val)-val)<1e-14 # I don't see how this holds for complex eri
            ERI[a,b,j,i] = val
            ERI[a,j,b,i] = val
            ERI[b,a,i,j] = val
            ERI[j,a,i,b] = val
            ERI[i,b,j,a] = val
            ERI[i,j,b,a] = val
            ERI[b,i,a,j] = val
            ERI[j,i,a,b] = val
            
        end
    end
    ERI/=2;

    (ERI,K,E0,NORB,NELEC,MS2)
end