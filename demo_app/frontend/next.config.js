/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow external images if needed
  images: {
    domains: [],
  },
  // Ensure proper handling of Plotly
  transpilePackages: ['react-plotly.js'],
}

module.exports = nextConfig
