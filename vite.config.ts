import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(({ command }) => {
  const isHmrDisabled = process.env.DISABLE_HMR === 'true';

  return {
    base: command === 'build' ? './' : '/',
    plugins: [react(), tailwindcss()],
    define: isHmrDisabled ? {
      'import.meta.hot': JSON.stringify({
        accept: () => {},
        dispose: () => {},
        prune: () => {},
        decline: () => {},
        invalidate: () => {},
        on: () => {},
      }),
    } : {},
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: !isHmrDisabled,
      watch: isHmrDisabled ? null : {},
    },
  };
});
