function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

function genPositiveArr!(arr::AbstractVector)
	s = map(arr) do e
		if e < 0
			return [0, abs(e)]
		else
			return [e, 0]
		end
	end

	return vcat(s...)
end
