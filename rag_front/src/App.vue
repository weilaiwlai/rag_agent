<script setup>
import { ref } from 'vue'
import { ChatLineRound, Collection } from '@element-plus/icons-vue'
import RagChat from './components/RagChat.vue'
import KnowledgeBaseViewer from './components/KnowledgeBaseViewer.vue'

const activeTab = ref('chat')
</script>

<template>
  <div class="app-container">
    <el-container class="main-layout">
      <el-header class="app-header">
        <h1 class="app-title">RAG 智能助手</h1>
        <el-menu
          :default-active="activeTab"
          class="navigation-menu"
          mode="horizontal"
          @select="(index) => activeTab = index"
          text-color="#333"
          active-text-color="#409EFF"
        >
          <el-menu-item index="chat">
            <el-icon><ChatLineRound /></el-icon>
            <span>智能问答</span>
          </el-menu-item>
          <el-menu-item index="knowledge">
            <el-icon><Collection /></el-icon>
            <span>知识库查看</span>
          </el-menu-item>
        </el-menu>
      </el-header>
      
      <el-main class="app-main">
        <RagChat v-if="activeTab === 'chat'" />
        <KnowledgeBaseViewer v-else-if="activeTab === 'knowledge'" />
      </el-main>
    </el-container>
  </div>
</template>

<style>
body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.app-container {
  height: 100vh;
}

.main-layout {
  height: 100%;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-bottom: 1px solid #ebeef5;
  z-index: 10;
}

.app-title {
  margin: 0;
  font-size: 24px;
  color: #303133;
}

.navigation-menu {
  border-bottom: none;
  border-right: none;
}

.app-main {
  padding: 0;
  height: calc(100%); 
  width: calc(100%);
}
</style>