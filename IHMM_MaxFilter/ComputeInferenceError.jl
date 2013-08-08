
using Debug
using PyCall
#require("HemmingDistance.py")
@pyimport HemmingDistance as hm
# @pyimport munkres as mk
# matrix = {{5, 9, 1},
# 	          {10, 3, 2},
# 	          {8, 7, 4}}
# t = hm.teest(matrix)

@debug begin


function encodeAlphabet(seq)
	alphabet = unique(seq)
	code_dict = Dict()
	index_to_alphabet_dict = Dict()
	for i = 1:length(alphabet)
		code_dict[alphabet[i]] = i
		index_to_alphabet_dict[i] = alphabet[i]
	end
	return {"code_dict" => code_dict, "index_to_alphabet_dict" => index_to_alphabet_dict}
end

function buildProfitMatrix(seq_true, seq_inferred)
	unique_inferred = unique(seq_inferred)
	unique_true = unique(seq_true)	
	profit_matrix = zeros(length(unique_inferred), length(unique_true))
	tru_dict = encodeAlphabet(seq_true)["code_dict"]
	inf_dict = encodeAlphabet(seq_inferred)["code_dict"]

	for inf in  unique_inferred
		for tru in unique_true
			replaced_indices = findin(seq_inferred, inf)
			for k in replaced_indices
				if seq_true[k] == tru
					profit_matrix[inf_dict[inf], tru_dict[tru]] += 1
				end
			end
		end
	end

	return profit_matrix

end


function computeError(seq_inferred, seq_true)
	matrix = buildProfitMatrix(seq_true, seq_inferred)
	indices = hm.teest(matrix)
	tru_dict = encodeAlphabet(seq_true)["index_to_alphabet_dict"]
	inf_dict = encodeAlphabet(seq_inferred)["index_to_alphabet_dict"]
	modified_indices = []
	# @bp
	for pair in indices
		encoded_index = pair[1] + 1
		target_encoded_index = pair[2] + 1
		alphabet = inf_dict[encoded_index]
		target_alphabet = tru_dict[target_encoded_index]
		for i = 1:length(seq_true)
			if seq_inferred[i] == alphabet && contains(modified_indices, i) == false
				seq_inferred[i] = target_alphabet
				modified_indices = vcat(modified_indices, i)
			end
		end
	end

	#println(seq_inferred)
	count_error = 0
	for t = 1:length(seq_true)
		if seq_true[t] != seq_inferred[t]
			count_error += 1
		end
	end
	return count_error
end



# seq_inferred = [1,1,1,3,3,3]
# seq_true = [1,1,3,3,4,4]

# matrix = buildProfitMatrix(seq_true, seq_inferred)


# indices = hm.teest(matrix)
# println(indices)
# inferred = computeError(indices, seq_inferred, seq_true)
# println(inferred)


end


