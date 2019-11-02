import read_write_objects

def main():

    output_filename = '../output/output.p'
    output = read_write_objects.read_object_from_file(output_filename)

    #print(output)

    for i in output:
        print(i)
        # for k, v in i:
        #     print(k)

if __name__ == '__main__':
    main()