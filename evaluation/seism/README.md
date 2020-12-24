# SEISM

The [SEISM](https://github.com/jponttuset/seism) repository is used to evaluate the edge detection task on PASCALContext.
A quick tutorial on how to set up seism with this repository is given below.

1) Clone the repository.
```shell
git clone https://github.com/jponttuset/seism
```

2) Run the install scripts (matlab is required).
```shell
matlab -nojvm -nodisplay -r "install"
```

3) Set the seism root dir in `./utils/mypath.py`.

4) Set the `db_root_dir` in `$SEISM_ROOT/src/gt_wrappers/db_root_dir.m`. The `db_root_dir` variable refers to the location where the edge ground truth annotations are stored. Mine looks as follows, since it only supports PASCALContext.

```matlab
function db_root_dir = db_root_dir( database )
db_root_dir='/path/to/PASCAL_MT/pascal-context/';
end
```

5) Copy the `./evaluation/seism/read_one_png.m` file to `$SEISM_ROOT/src/io/`.

6) In order to speed up the evaluation, the parallel toolbox can be used in the `$SEISM_ROOT/src/scripts/eval_method_all_params.m` file. The sequential evaluation can be commented out. Mine looks as follows:

```matlab
%% Run using the parallel computing toolbox
parfor nn=1:length(experiments)
     method_name = experiments(nn).method;
     parameter   = experiments(nn).parameter;
     measure     = experiments(nn).measure;
     disp(['Starting: ' method_name ' (' parameter ') for measure ' measure ' on ' gt_set])
     eval_method(method_name, parameter, measure, read_part_fun, database,  gt_set, length(params), segm_or_contour,cat_ids)
     disp(['Done:     ' method_name ' (' parameter ') for measure ' measure ' on ' gt_set])
 end
```
When the `evaluation/eval_edge.py` is used, a custom matlab script will be generated and executed using MATLAB.
