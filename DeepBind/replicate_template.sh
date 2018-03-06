for dir in $(ls -d D*); do
    #echo $dir
    cd /users/jisraeli/.kipoi/models/tf-binding/DeepBind/$dir
    rm dataloader.py
    ln -s ../template/dataloader.py dataloader.py

    rm dataloader.yaml
    ln -s ../template/dataloader.yaml dataloader.yaml

    rm model.yaml
    ln -s ../template/model.yaml model.yaml

    rm examples_files
    ln -s ../template/example_files/ examples_files

    rm custom_keras_objects.py
    ln -s ../template/custom_keras_objects.py custom_keras_objects.py
done
