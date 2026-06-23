import pickle
SIM_prefix_str = "[SIMULATE_HBM_DEBUG]"
SIM_frame_prefix_str = " [SIMULATE_HBM_DEBUG]Frames:SIMULATE_HBM_DEBUG Frames"
SIM_frame_star_str = "Python stack start"
SIM_frame_end_str = "Python stack end"
SIM_Node_prefix_str = "[SIMULATE_HBM_DEBUG]NODE UniqueId"


def parse_frame_line(line_str):
    # the line template:
    # my_func (test.py:10)
    filename = "None"
    line = "None"
    name = "None"

    line_str = line_str.strip()
    line_str_split = line_str.split("(")
    name = line_str_split[0].strip()

    if len(line_str_split)>= 2 :
        line_str_split = line_str_split[1].split(":")
        filename = line_str_split[0]

        if len(line_str_split)>= 2 :
            line = line_str_split[1][:-1]

    return filename, line, name

def remove_trace_with_zero_size(data):
    data_select = {}
    data_select['segments'] = []
    data_select['device_traces']= []
    device_traces_0 = []
    data_select['device_traces'].append(device_traces_0)
    for i in range(len(data['device_traces'][0])):
        if data['device_traces'][0][i]['size'] >0 :
            data_select['device_traces'][0].append(data['device_traces'][0][i])
    return data_select

def parse_log(filename):
    data = {}
    data['device_traces']=[]

    curr_gcu_traces = []
    f = open(filename, encoding="utf8", errors='ignore')
    line = f.readline()
    while line:
        # start to parse log for one node alloc or node free
        if SIM_Node_prefix_str in line:
            element = {}

            # set stream tobe 0 as default
            element["stream"] = '0'

            # get uniqueId of this node, and use uniqueId as "addr" while ploting trace.html
            line_split = line.split(":")
            uniqueId = line_split[-1].strip()
            element["addr"] = uniqueId
            line = f.readline()

            # get node name and append it into frames, in order to display name while ploting trace.html
            line_split = line.split(":")
            node_name = line_split[-1].strip()
            element["node_name"] = node_name
            line = f.readline()

            element["frames"] = []
            frame = {}
            frame['filename'] = node_name
            frame['line'] = "None"
            frame['name'] = "None"
            element["frames"].append(frame)

            # get action type, and use it as "action" while ploting trace.html
            line_split = line.split(":")
            element["size"] = int(line_split[-1][:-6])
            if "Alloc" in line_split[-2]:
                element["action"] = 'alloc'
            elif "Free" in line_split[-2]:
                element["action"] = 'free_completed'
            else:
                element["action"] = 'unkown'
                print("donot support this kind of action.")
            line = f.readline()

            # get frames
            while line:
                if "Python stack start" in line:
                    line = f.readline()
                    break
                line = f.readline()
            while line:
                if "Python stack end" in line:
                    break
                (filename, line, name) = parse_frame_line(line)
                frame = {}
                frame['filename'] = filename
                frame['line'] = line
                frame['name'] = name
                element["frames"].append(frame)
                line = f.readline()

            curr_gcu_traces.append(element)
        line = f.readline()

    f.close()
    data['device_traces'].append(curr_gcu_traces)

    data = remove_trace_with_zero_size(data)
    return data


def grep_log(input_filename, output_filename):
    input_f = open(input_filename, 'r', encoding="utf8", errors='ignore')
    output_f = open(output_filename, 'w', encoding="utf8", errors='ignore')
    line = input_f.readline()
    while line:
        if SIM_frame_prefix_str in line:
            output_f.write(line)
            while line:
                if SIM_frame_star_str in line:
                    output_f.write(line)
                    line = input_f.readline()
                    break
                line = input_f.readline()
            while line:
                output_f.write(line)
                if SIM_frame_end_str in line:
                    break
                line = input_f.readline()
        if (SIM_prefix_str in line) and (SIM_frame_prefix_str not in line):
            output_f.write(line)

        line = input_f.readline()
    input_f.close()
    output_f.close()
    return


if __name__ == "__main__":
    import sys
    data = parse_log(sys.argv[1])

    import pickle
    pickle.dump(data, open(sys.argv[2], 'wb'))

