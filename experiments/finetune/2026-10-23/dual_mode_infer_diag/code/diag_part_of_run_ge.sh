  diag)   # inference A/B (2026-10-23): U p8 E outputs did not keep the given video (known MAE 35-55) and G did not follow the control
    cd $X/lora_ge; E="E_real_6662a9fd_da E_real_55494490_da"; G="G_real_55494490_t"
    $PY predict_ge.py --part ev_U_8 --cases --train_rows real_23e9d00a real_edf79415 --tag dgU8_train || exit 1          # same settings as the eval, TRAINING rows
    $PY predict_ge.py --part ev_U_8 --cases --train_rows real_23e9d00a --guidance 1 --tag dgU8_train_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --guidance 1 --tag dgU8_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --guidance 6 --negative empty --tag dgU8_negempty || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $E --no_lora --guidance 1 --tag dgbase_cfg1 || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $G --guidance 6 --tag dgU8_ref || exit 1
    $PY predict_ge.py --part ev_U_8 --cases $G --guidance 1 --no_ref_g --tag dgU8_gcfg1 || exit 1 ;;
