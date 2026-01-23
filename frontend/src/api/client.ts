import axios from 'axios';

const client = axios.create({
  baseURL: 'http://localhost:8080/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getPrice = async (symbol: string) => {
  const response = await client.get(`/market/price/${symbol}`);
  return response.data;
};

export default client;
