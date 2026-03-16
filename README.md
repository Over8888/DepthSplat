# DepthSplat v3 Frontend

A React + TypeScript + Vite frontend for a remote DepthSplat inference backend.

## Folder structure

```text
src/
  components/
  hooks/
  pages/
  services/
  store/
  styles/
  types/
  utils/
```

## Notes

- The frontend only uses the provided backend endpoints.
- Because no `GET /tasks` endpoint exists in the contract, the History page uses locally persisted task history from tasks created/opened in this browser.
- Backend task state is the source of truth.
- `cancelled` is handled distinctly from `failed`.

## Run

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Manual validation checklist

1. Create a task with uploaded images.
2. Confirm redirect to `/tasks/:id`.
3. Watch task status polling and live logs.
4. Cancel a running task and confirm the UI waits until backend state is `cancelled`.
5. Verify success shows full results.
6. Verify failed tasks show error summary + logs.
7. Verify cancelled tasks show banner and partial results only if returned.
8. Verify history filters by state, sample, and time.
9. Verify settings persist backend URL and defaults in localStorage.
