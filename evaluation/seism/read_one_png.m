function partition = read_one_cont_png(method_dir, parameter, im_id)
    res_file = fullfile(method_dir, [im_id '.png']);
    if ~exist(res_file,'file')
        partition = [];
    else
        partition = double((double(imread(res_file))/255.0) >= str2double(parameter));
    end
end
