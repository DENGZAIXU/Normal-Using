import json
import os

def json2json(file_path, file_name):
    data = json.load(open(os.path.join(file_path,'annotation',file_name)))

    fname, ext = os.path.splitext(file_name)
    str = fname + ".jpg272893"
    data = data[str]

    save_file = os.path.join(file_path, 'ann/',file_name)

    new_data = {}

    ## 首先读取 Fang的json文件，在生成新的json文件
    data_x = data["regions"][0]["shape_attributes"]['x']
    data_y = data["regions"][0]["shape_attributes"]['y']
    data_w = data["regions"][0]["shape_attributes"]['width']
    data_h = data["regions"][0]["shape_attributes"]['height']

    new_data["version"] = "3.16.1"
    new_data["flags"] = {}
    new_data["shapes"] = []
    new_data["lineColor"] = [0, 255, 0, 128]
    new_data["fillColor"] = [255, 255, 0, 64]

    new_data["imagePath"] = fname + ".jpg"
    new_data["imageData"] = None
    new_data["imageWidth"] = 1920
    new_data["imageHeight"] = 1080

    cont = {}
    cont["label"] = "ww"
    cont["line_color"] = None
    cont["fill_color"] = None
    cont["points"] = [[data_x, data_y], [data_x + data_w, data_y + data_h]]
    cont["shape_type"] = "rectangle"
    cont["flags"] = {}
    new_data["shapes"].append(cont)

    with open(save_file,'w+') as f:
        json.dump(new_data, f,indent= 4)

def main():
    data_path = r"C:\Users\Xu\Desktop\data\wz"
    # image_path = os.path.join(data_path,"image")
    ann_path = os.path.join(data_path,"annotation")
    all_ann = os.listdir(ann_path)
    print(len(all_ann))
    for file_path in all_ann:
        json2json(data_path,file_path)


if __name__ == '__main__':
    main()
