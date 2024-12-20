struct fast_init{T, S, N₁, N₂, A}
    space::TensorMapSpace{S,N₁,N₂}
    totaldim::Int

    function fast_init(codom::ProductSpace{S,N₁},
        dom::ProductSpace{S,N₂},stortype) where {S<:IndexSpace,N₁,N₂}

        homspace = codom←dom
        return new{eltype(stortype),S,N₁,N₂,stortype}(homspace,TensorKit.fusionblockstructure(homspace).totaldim)
    end

    function (d::fast_init{T, S, N₁, N₂, A})(alloc=TensorOperations.DefaultAllocator(),istemp=Val(false)) where {T,S, N₁, N₂, A<:DenseVector}
        data = tensoralloc(A, d.totaldim, istemp,alloc)
        return TensorMap{T,S,N₁,N₂,A}(data, d.space)
    end

end

function TensorOperations.tensorfree!(a::AbstractArray, ::TensorOperations.DefaultAllocator)
end

function subsplit(ex,instantiated_struct_name)
    a = Any[ex.head]
    b = Any[ex.head]
    c = []
    for (sa,sb,sc) in (split_execution(a,instantiated_struct_name) for a in ex.args)
        append!(c,sc)
        push!(a,sa)
        push!(b,sb)
    end
    return (Expr(a...),Expr(b...),c)
end

function split_execution(ex,instantiated_struct_name)
    splitmap = Dict(GlobalRef(TensorOperations,:tensorcontract!) => (create_mediated_tensorcontract!,mediated_tensorcontract!),
                    GlobalRef(TensorOperations,:tensoralloc_contract) => (create_mediated_tensoralloc_contract,mediated_tensoralloc_contract),
                    GlobalRef(TensorOperations,:tensoradd!) => (create_mediated_tensoradd!,mediated_tensoradd!),
                    GlobalRef(TensorOperations,:tensoralloc_add) => (create_mediated_tensoralloc_add,mediated_tensoralloc_add),
                    GlobalRef(TensorOperations,:tensortrace!) => (create_mediated_tensortrace!,mediated_tensortrace!),
                    
                    GlobalRef(TensorKit,:planarcontract!) => (create_mediated_planarcontract!,mediated_planarcontract!),
                    GlobalRef(TensorKit,:planaradd!) => (create_mediated_planaradd!,mediated_planaradd!),
                    GlobalRef(TensorKit,:planartrace!) => (create_mediated_planartrace!,mediated_planartrace!))

    if !(ex isa Expr)
        return (ex,ex,[])
    elseif ex.head == :(=) && length(ex.args) == 2
        if ex.args[2] isa Expr && ex.args[2].head == :call
            t = ex.args[2].args[1]

            if t in keys(splitmap)
                (mapped_1,mapped_2) = splitmap[t]
                nvar = gensym()
                a = quote
                    ($(ex.args[1]),$(nvar)) = $(mapped_1)($(ex.args[2].args[2:end]...))
                end
                b = quote
                    $(ex.args[1]) = $(mapped_2)($instantiated_struct_name,$(nvar),$(ex.args[2].args[2:end]...))
                end
                return (a,b,[nvar])
            end
        end

        return subsplit(ex,instantiated_struct_name)
    else# ex.head in (:block,)
        return subsplit(ex,instantiated_struct_name)
    end
end
split_execution(ex::Symbol,instantiated_struct_name) = (ex,ex,[])



macro tightloop_tensor(name,args::Vararg{Expr})
    isempty(args) && throw(ArgumentError("No arguments passed to `@tensor`"))
    
    allocator = TensorOperations.DefaultAllocator();
    #if length(args) == 1
    #    parser = TensorOperations.defaultparser
    #else
        tensorexpr = args[end]
        kwargs = TensorOperations.parse_tensor_kwargs(args[1:(end - 1)])
        parser = TensorOperations.tensorparser(tensorexpr, kwargs...)
        for (name,val) in kwargs
            if name == :allocator
                allocator = eval(val)
            end
        end
    #end
    
    parsed = parser(tensorexpr)
    
    instantiated_struct_name = gensym()
    (a,b,c) = split_execution(parsed,instantiated_struct_name)
    c_types = [gensym() for t in c]
    declaration = quote end
    for (c_v,c_t) in zip(c,c_types)
        declaration = quote
            $(declaration)
            $(c_v)::$(c_t)
        end
    end

    input_symbols =  TensorOperations.getinputtensorobjects(args[end])
    output_symbols =  TensorOperations.getoutputtensorobjects(args[end])
    
    arg_symbols = [input_symbols...,output_symbols...];
    kwarg_expr = Expr(:parameters,[Expr(:kw,s,nothing) for s in arg_symbols]...)
    abstract_eval_call = Expr(:parameters,[Expr(:kw,s,Expr(:call,GlobalRef(MPSKitExperimental,:SymbolicTensorMap),Expr(:call,:getindex,s,1),Expr(:call,:getindex,s,2))) for s in arg_symbols]...)

    access_inner_fields = quote end
    for c_v in c
        access_inner_fields = quote
            $access_inner_fields
            $(c_v) = $(instantiated_struct_name).$(c_v)
        end
    end

    return esc(quote
        struct $(name){A,$(c_types...)}
            allocator::A
            $(declaration)
            
            function $(name)($(kwarg_expr))
                tup = abstract_eval($(abstract_eval_call))
                new{typeof($(allocator)),typeof.(tup)...}($(allocator),tup...)
            end
            
            function abstract_eval($(kwarg_expr))
                $(a)
                return tuple($(c...))
            end
            function ($(instantiated_struct_name)::$name)($(kwarg_expr))
                $(access_inner_fields)
                $(b)
            end
        end
    end)
end


macro tightloop_planar(name,args::Vararg{Expr})
    isempty(args) && throw(ArgumentError("No arguments passed to `@planar`"))
    
    allocator = TensorOperations.DefaultAllocator();

    tensorexpr = args[end]
    kwargs = TensorOperations.parse_tensor_kwargs(args[1:(end - 1)])
    parser = TensorKit.planarparser(tensorexpr, kwargs...)
    for (name,val) in kwargs
        if name == :allocator
            allocator = eval(val)
            #allocator =  TensorOperations.Backend{val}()
        end
    end
    
    parsed = parser(tensorexpr)
    
    instantiated_struct_name = gensym()
    (a,b,c) = split_execution(parsed,instantiated_struct_name)
    c_types = [gensym() for t in c]
    declaration = quote end
    for (c_v,c_t) in zip(c,c_types)
        declaration = quote
            $(declaration)
            $(c_v)::$(c_t)
        end
    end

    input_symbols =  TensorOperations.getinputtensorobjects(args[end])
    output_symbols =  TensorOperations.getoutputtensorobjects(args[end])
    arg_symbols = unique([input_symbols...,output_symbols...])
    kwarg_expr = Expr(:parameters,[Expr(:kw,s,nothing) for s in arg_symbols]...)
    abstract_eval_call = Expr(:parameters,[Expr(:kw,s,Expr(:call,GlobalRef(MPSKitExperimental,:SymbolicTensorMap),Expr(:call,:getindex,s,1),Expr(:call,:getindex,s,2))) for s in arg_symbols]...)

    access_inner_fields = quote end
    for c_v in c
        access_inner_fields = quote
            $access_inner_fields
            $(c_v) = $(instantiated_struct_name).$(c_v)
        end
    end

    return esc(quote
        struct $(name){A,$(c_types...)}
            allocator::A
            $(declaration)
            
            function $(name)($(kwarg_expr))
                tup = abstract_eval($(abstract_eval_call))
                new{typeof($(allocator)),typeof.(tup)...}($(allocator),tup...)
            end
            
            function abstract_eval($(kwarg_expr))
                $(a)
                return tuple($(c...))
            end
            function ($(instantiated_struct_name)::$name)($(kwarg_expr))
                $(access_inner_fields)
                $(b)
            end
        end
    end)
end
