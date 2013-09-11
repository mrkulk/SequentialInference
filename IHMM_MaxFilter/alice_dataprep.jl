# #Alice in Wonderland - CH1

# function get_AW_dataset(seed)
# 	fp = open("alice_in_wonderland.txt")
# 	data = readall(fp)
# 	data = lowercase(data)
# 	data=split(data,"")

# 	LENGTH_SEQ = 500
# 	hidden_state_seq = zeros(LENGTH_SEQ)
# 	observation_seq = zeros(LENGTH_SEQ)

# 	tindx=1
# 	test_dict = Dict()
# 	for i=1:length(data)
# 		if 	data[i] != "!" && data[i] != "'" && data[i] != "—" && data[i] != "?" &&
# 			data[i] != "-" && data[i] != "." && data[i] != " " && data[i] != "\n" && 
# 			data[i] != "," && data[i] != ")" && data[i] != "(" && data[i] != ":" && data[i] != ";"      
			
# 				if tindx <= LENGTH_SEQ
# 					observation_seq[tindx] = data[i][1] - 97 + 1
# 					hidden_state_seq[tindx] = -1
# 					tindx += 1
# 					test_dict[data[i]] = 1
# 				else
# 					break
# 				end
# 		end
# 	end
	
# 	# println(test_dict) 
# 	# println(length(test_dict))


# #	NUM_OBS = length(test_dict)
# NUM_OBS = 26
# 	return {"hid" => hidden_state_seq, "obs" => observation_seq}, NUM_OBS

# end


# get_AW_dataset(0)

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
<<<<<<< HEAD
	for i=_start:_end#length(data)
=======
	for i=1:LENGTH_SEQ#length(data)
>>>>>>> 129634c0c2fd332d6c98b1fa729b28950bc33b33
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
	
	observation_seq = observation_seq[1:tindx-1]
	hidden_state_seq = hidden_state_seq[1:tindx - 1]
	# println(test_dict) 
	# println(length(test_dict))
	println(length(observation_seq))
	println(sort(unique(observation_seq)))
	println(length(unique(observation_seq)))
	NUM_OBS = 26
	return {"hid" => hidden_state_seq, "obs" => observation_seq}, NUM_OBS

end


<<<<<<< HEAD
get_AW_dataset(0, 1, 2000)
=======
get_AW_dataset(0)
>>>>>>> 129634c0c2fd332d6c98b1fa729b28950bc33b33
