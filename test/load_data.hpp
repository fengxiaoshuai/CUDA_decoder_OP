#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

void load(vector<float>& weight, string dir)
{
        ifstream input(dir);
        if (input.fail())
        {
                cout << "File does not exist" << endl;
                cout << "Exit program" << endl;
                return;
        }
        float num=0.0;
        while (input>>num)  // 当没有读到文件结尾
        {
                weight.push_back(num);
                //cout << num << endl;
        }
        input.close();
}

void load_layer_weight(vector<vector<float>>& layer_weight, int num)
{
        cout << "start read layer " << num << " weight" << endl;
        vector<float> layer_self_scale;//0
        vector<float> layer_self_bias;//1
        vector<float> layer_self_q;//2
        vector<float> layer_self_k;//3
        vector<float> layer_self_v;//4
        vector<float> layer_self_last;//5

        vector<float> layer_encdec_scale;//6
        vector<float> layer_encdec_bias;//7
        vector<float> layer_encdec_q;//8
        vector<float> layer_encdec_k;//cache k//9
        vector<float> layer_encdec_v;//cache v//10
        vector<float> layer_encdec_last;//11

        vector<float> layer_ffn_scale;//12
        vector<float> layer_ffn_bias;//13
        vector<float> layer_ffn_first_weight;//14
        vector<float> layer_ffn_first_bias;//15
        vector<float> layer_ffn_second_weight;//16
        vector<float> layer_ffn_second_bias;//17

        vector<float> layer_self_position_key;//18
        vector<float> layer_self_position_value;//19


        cout << "...:load self attention weight" << endl;
        string name = "./weight/layer_" + to_string(num) ;
        load(layer_self_scale, name + "_self_scale.txt");
        load(layer_self_bias, name + "_self_bias.txt");
        load(layer_self_q, name + "_self_q.txt");
        load(layer_self_k, name + "_self_k.txt");
        load(layer_self_v, name + "_self_v.txt");
        load(layer_self_last, name + "_self_last.txt");
        load(layer_self_position_key, name + "_self_position_key.txt");
        load(layer_self_position_value, name + "_self_position_value.txt");
        cout << "...:load encdec attention weight" << endl;
        load(layer_encdec_scale, name + "_encdec_scale.txt");
        load(layer_encdec_bias, name + "_encdec_bias.txt");
        load(layer_encdec_q, name + "_encdec_q.txt");
        load(layer_encdec_k, name + "_encdec_k.txt");
        load(layer_encdec_v, name + "_encdec_v.txt");
        load(layer_encdec_last, name + "_encdec_last.txt");
        cout << "...:load read fnn weight" << endl;
        load(layer_ffn_scale, name + "_ffn_scale.txt");
        load(layer_ffn_bias, name + "_ffn_bias.txt");
        load(layer_ffn_first_weight, name + "_ffn_first_weight.txt");
        load(layer_ffn_first_bias, name + "_ffn_first_bias.txt");
        load(layer_ffn_second_weight, name + "_ffn_second_weight.txt");
        load(layer_ffn_second_bias, name + "_ffn_second_bias.txt");


        layer_weight.push_back(layer_self_scale);
        layer_weight.push_back(layer_self_bias);
        layer_weight.push_back(layer_self_q);
        layer_weight.push_back(layer_self_k);
        layer_weight.push_back(layer_self_v);
        layer_weight.push_back(layer_self_last);

        layer_weight.push_back(layer_encdec_scale);
        layer_weight.push_back(layer_encdec_bias);
        layer_weight.push_back(layer_encdec_q);
        layer_weight.push_back(layer_encdec_k);
        layer_weight.push_back(layer_encdec_v);
        layer_weight.push_back(layer_encdec_last);
        layer_weight.push_back(layer_ffn_scale);
        layer_weight.push_back(layer_ffn_bias);
        layer_weight.push_back(layer_ffn_first_weight);
        layer_weight.push_back(layer_ffn_first_bias);
        layer_weight.push_back(layer_ffn_second_weight);
        layer_weight.push_back(layer_ffn_second_bias);

        layer_weight.push_back(layer_self_position_key);
        layer_weight.push_back(layer_self_position_value);

        cout << "...:end layer " << num << " weight" << endl;
}
