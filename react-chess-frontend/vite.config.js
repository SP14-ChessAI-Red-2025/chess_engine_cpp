import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/website/', // Set this to the subdirectory of your GitHub Pages site
});
