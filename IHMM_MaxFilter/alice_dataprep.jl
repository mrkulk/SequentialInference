#Alice in Wonderland - CH1

function get_AW_dataset(seed, _start, _end)
	fp = open("alice_in_wonderland.txt")
	data = readall(fp)
	data = lowercase(data)
	data=split(data,"")

	LENGTH_SEQ = _end - _start#5000
	hidden_state_seq = zeros(LENGTH_SEQ)
	observation_seq = zeros(LENGTH_SEQ)

	tindx=1
	test_dict = Dict()
	for i=1:LENGTH_SEQ#length(data)
		if 	data[i] != "!" && data[i] != "'" && data[i] != "—" && data[i] != "?" &&
			data[i] != "-" && data[i] != "." && data[i] != " " && data[i] != "\n" && 
			data[i] != "," && data[i] != ")" && data[i] != "(" && data[i] != ":" && data[i] != ";"      
			
				if tindx <= LENGTH_SEQ
					observation_seq[tindx] = data[i][1] - 97 + 1
					hidden_state_seq[tindx] = -1
					tindx += 1
					test_dict[data[i]] = 1
				else
					break
				end
		end
	end
	
	# println(test_dict) 
	# println(length(test_dict))
	# println(observation_seq)

	NUM_OBS = length(test_dict)
	return {"hid" => hidden_state_seq, "obs" => observation_seq}, NUM_OBS

end


get_AW_dataset(0)
