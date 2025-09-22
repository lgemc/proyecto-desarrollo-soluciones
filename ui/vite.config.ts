import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    allowedHosts: ['animalitos-front.tunn.dev']
  },
  define: {
    __API_URL__: JSON.stringify(process.env.VITE_API_URL || '/api/v1')
  }
})