# DepthSplat v3 Linux Backend

单机、单 GPU、基于 subprocess 的 DepthSplat 推理后端。

## 项目结构

```text
app/
  main.py
  api/
    routes/
  services/
    task_manager.py
    runner.py
    result_builder.py
    sample_service.py
    storage.py
  models/
  schemas/
  utils/
  config.py
```

## 任务目录

```text
/root/autodl-tmp/demo_v2/outputs/tasks/<task_id>/
  input/
  render/
  depth/
  logs/
    stdout.log
    stderr.log
  meta/
    request.json
    result.json
    timing.json
    cancel.json
    task.json
```

## API

- `GET /samples`
- `GET /presets`
- `POST /tasks`
- `GET /tasks/{id}`
- `POST /tasks/{id}/cancel`
- `GET /tasks/{id}/logs`
- `GET /tasks/{id}/result`
- `GET /healthz`

结果静态文件通过：

- `GET /artifacts/{task_id}/...`

## 运行

```bash
cd /root/autodl-tmp/demo_v2
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8012
```

或：

```bash
bash /root/autodl-tmp/demo_v2/scripts/run_backend.sh
```

如需临时改端口，也可以：

```bash
DEPTHSPLAT_V3_PORT=8012 bash /root/autodl-tmp/demo_v2/scripts/run_backend.sh
```

## 说明

- 单机串行执行，一次只跑一个任务。
- backend state 使用：
  - `queued`
  - `preparing`
  - `running`
  - `postprocessing`
  - `success`
  - `failed`
  - `cancelled`
- `result.json` 是前端读取结果的唯一标准契约，前端不应直接扫描目录。

## 最小验证

```bash
cd /root/autodl-tmp/demo_v2
python scripts/validate_backend.py
```
