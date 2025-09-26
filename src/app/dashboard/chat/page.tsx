'use client';

import React, { useState, useEffect, useRef } from 'react';
import CommunityForum from '@/components/CommunityForum';

interface ChatMessage {
  id: string;
  userId: string;
  username: string;
  avatar?: string;
  message: string;
  timestamp: string;
  images?: string[];
  likes: number;
  replies: ChatMessage[];
  isTradeIdea?: boolean;
  symbol?: string;
  tradeType?: 'long' | 'short';
  entryPrice?: number;
  targetPrice?: number;
  stopLoss?: number;
}

interface ChatRoom {
  id: string;
  name: string;
  description: string;
  memberCount: number;
  isActive: boolean;
}

export default function ChatPage() {
  const [selectedRoom, setSelectedRoom] = useState<string>('general');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState<string>('');
  const [isTradeIdea, setIsTradeIdea] = useState<boolean>(false);
  const [tradeDetails, setTradeDetails] = useState({
    symbol: '',
    tradeType: 'long' as 'long' | 'short',
    entryPrice: '',
    targetPrice: '',
    stopLoss: ''
  });
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const chatRooms: ChatRoom[] = [
    { id: 'general', name: 'General Discussion', description: 'General trading discussion', memberCount: 1247, isActive: true },
    { id: 'stocks', name: 'Stock Trading', description: 'Stock market discussions', memberCount: 892, isActive: true },
    { id: 'crypto', name: 'Cryptocurrency', description: 'Crypto trading and analysis', memberCount: 1456, isActive: true },
    { id: 'options', name: 'Options Trading', description: 'Options strategies and ideas', memberCount: 634, isActive: true },
    { id: 'forex', name: 'Forex', description: 'Foreign exchange trading', memberCount: 423, isActive: true },
    { id: 'ideas', name: 'Trade Ideas', description: 'Share your trade setups', memberCount: 789, isActive: true }
  ];

  useEffect(() => {
    // Mock messages - replace with actual WebSocket connection
    const mockMessages: ChatMessage[] = [
      {
        id: '1',
        userId: 'user1',
        username: 'TradeMaster2024',
        message: 'AAPL looking strong after earnings. Anyone else seeing this breakout?',
        timestamp: '2024-01-15T10:30:00Z',
        likes: 12,
        replies: [],
        isTradeIdea: true,
        symbol: 'AAPL',
        tradeType: 'long',
        entryPrice: 185.50,
        targetPrice: 195.00,
        stopLoss: 180.00
      },
      {
        id: '2',
        userId: 'user2',
        username: 'CryptoKing',
        message: 'Bitcoin forming a nice cup and handle pattern. Could see $50k soon! 🚀',
        timestamp: '2024-01-15T10:35:00Z',
        likes: 8,
        replies: [
          {
            id: '2-1',
            userId: 'user3',
            username: 'BTCAnalyst',
            message: 'I agree! Volume is confirming the breakout.',
            timestamp: '2024-01-15T10:37:00Z',
            likes: 3,
            replies: []
          }
        ]
      },
      {
        id: '3',
        userId: 'user4',
        username: 'OptionsGuru',
        message: 'Check out this TSLA setup I\'m watching',
        timestamp: '2024-01-15T10:40:00Z',
        images: ['/api/placeholder/400/300'],
        likes: 15,
        replies: [],
        isTradeIdea: true,
        symbol: 'TSLA',
        tradeType: 'long',
        entryPrice: 240.00,
        targetPrice: 260.00,
        stopLoss: 230.00
      }
    ];
    setMessages(mockMessages);
  }, [selectedRoom]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (!newMessage.trim() && selectedImages.length === 0) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      userId: 'currentUser',
      username: 'You',
      message: newMessage,
      timestamp: new Date().toISOString(),
      images: selectedImages.map(file => URL.createObjectURL(file)),
      likes: 0,
      replies: [],
      isTradeIdea,
      ...(isTradeIdea && {
        symbol: tradeDetails.symbol,
        tradeType: tradeDetails.tradeType,
        entryPrice: parseFloat(tradeDetails.entryPrice),
        targetPrice: parseFloat(tradeDetails.targetPrice),
        stopLoss: parseFloat(tradeDetails.stopLoss)
      })
    };

    setMessages(prev => [...prev, message]);
    setNewMessage('');
    setSelectedImages([]);
    setIsTradeIdea(false);
    setTradeDetails({
      symbol: '',
      tradeType: 'long',
      entryPrice: '',
      targetPrice: '',
      stopLoss: ''
    });
  };

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedImages(prev => [...prev, ...files].slice(0, 4)); // Max 4 images
  };

  const removeImage = (index: number) => {
    setSelectedImages(prev => prev.filter((_, i) => i !== index));
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Sidebar - Chat Rooms */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Trading Chat</h2>
          <p className="text-sm text-gray-400">Connect with fellow traders</p>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-2">
            {chatRooms.map((room) => (
              <button
                key={room.id}
                onClick={() => setSelectedRoom(room.id)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedRoom === room.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{room.name}</div>
                    <div className="text-xs text-gray-400">{room.description}</div>
                  </div>
                  <div className="text-xs">
                    <div className="flex items-center space-x-1">
                      <div className={`w-2 h-2 rounded-full ${room.isActive ? 'bg-green-400' : 'bg-gray-500'}`}></div>
                      <span>{room.memberCount}</span>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <div className="bg-gray-800 border-b border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">
                {chatRooms.find(room => room.id === selectedRoom)?.name}
              </h3>
              <p className="text-sm text-gray-400">
                {chatRooms.find(room => room.id === selectedRoom)?.memberCount} members online
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <button className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className="flex space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-semibold text-sm">
                  {message.username.charAt(0).toUpperCase()}
                </span>
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-1">
                  <span className="font-semibold text-white">{message.username}</span>
                  <span className="text-xs text-gray-400">{formatTime(message.timestamp)}</span>
                  {message.isTradeIdea && (
                    <span className="bg-green-600 text-white px-2 py-1 rounded text-xs">
                      Trade Idea
                    </span>
                  )}
                </div>
                
                {message.isTradeIdea && message.symbol && (
                  <div className="bg-gray-800 rounded-lg p-3 mb-2 border-l-4 border-green-500">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-white">{message.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        message.tradeType === 'long' ? 'bg-green-600' : 'bg-red-600'
                      }`}>
                        {message.tradeType?.toUpperCase()}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div>
                        <div className="text-gray-400">Entry</div>
                        <div className="text-white">${message.entryPrice?.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400">Target</div>
                        <div className="text-green-400">${message.targetPrice?.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-gray-400">Stop Loss</div>
                        <div className="text-red-400">${message.stopLoss?.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                )}
                
                <p className="text-gray-300 mb-2">{message.message}</p>
                
                {message.images && message.images.length > 0 && (
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    {message.images.map((image, index) => (
                      <img
                        key={index}
                        src={image}
                        alt={`Attachment ${index + 1}`}
                        className="rounded-lg max-w-full h-32 object-cover cursor-pointer hover:opacity-80 transition-opacity"
                      />
                    ))}
                  </div>
                )}
                
                <div className="flex items-center space-x-4 text-sm text-gray-400">
                  <button className="flex items-center space-x-1 hover:text-white transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                    <span>{message.likes}</span>
                  </button>
                  <button className="hover:text-white transition-colors">Reply</button>
                </div>
                
                {message.replies.length > 0 && (
                  <div className="mt-3 ml-4 space-y-2">
                    {message.replies.map((reply) => (
                      <div key={reply.id} className="flex space-x-2">
                        <div className="w-6 h-6 bg-gray-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-white text-xs">
                            {reply.username.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="font-medium text-white text-sm">{reply.username}</span>
                            <span className="text-xs text-gray-400">{formatTime(reply.timestamp)}</span>
                          </div>
                          <p className="text-gray-300 text-sm">{reply.message}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input Area */}
        <div className="bg-gray-800 border-t border-gray-700 p-4">
          {/* Trade Idea Toggle */}
          <div className="mb-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={isTradeIdea}
                onChange={(e) => setIsTradeIdea(e.target.checked)}
                className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-300">Share as Trade Idea</span>
            </label>
          </div>

          {/* Trade Idea Details */}
          {isTradeIdea && (
            <div className="mb-3 p-3 bg-gray-700 rounded-lg">
              <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                <input
                  type="text"
                  placeholder="Symbol"
                  value={tradeDetails.symbol}
                  onChange={(e) => setTradeDetails(prev => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
                  className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <select
                  value={tradeDetails.tradeType}
                  onChange={(e) => setTradeDetails(prev => ({ ...prev, tradeType: e.target.value as 'long' | 'short' }))}
                  className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="long">Long</option>
                  <option value="short">Short</option>
                </select>
                <input
                  type="number"
                  placeholder="Entry Price"
                  value={tradeDetails.entryPrice}
                  onChange={(e) => setTradeDetails(prev => ({ ...prev, entryPrice: e.target.value }))}
                  className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="number"
                  placeholder="Target Price"
                  value={tradeDetails.targetPrice}
                  onChange={(e) => setTradeDetails(prev => ({ ...prev, targetPrice: e.target.value }))}
                  className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <input
                  type="number"
                  placeholder="Stop Loss"
                  value={tradeDetails.stopLoss}
                  onChange={(e) => setTradeDetails(prev => ({ ...prev, stopLoss: e.target.value }))}
                  className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          )}

          {/* Selected Images Preview */}
          {selectedImages.length > 0 && (
            <div className="mb-3 flex space-x-2">
              {selectedImages.map((file, index) => (
                <div key={index} className="relative">
                  <img
                    src={URL.createObjectURL(file)}
                    alt={`Preview ${index + 1}`}
                    className="w-16 h-16 object-cover rounded-lg"
                  />
                  <button
                    onClick={() => removeImage(index)}
                    className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 rounded-full flex items-center justify-center text-white text-xs hover:bg-red-700"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Message Input */}
          <div className="flex items-end space-x-2">
            <div className="flex-1">
              <textarea
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Share your thoughts, analysis, or trade ideas..."
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                rows={2}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
            </div>
            
            <div className="flex space-x-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
                title="Attach images"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
              
              <button
                onClick={handleSendMessage}
                disabled={!newMessage.trim() && selectedImages.length === 0}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}