<template>
  <div class="knowledge-base-viewer">
    <!-- 顶部导航 -->
    <header class="viewer-header">
      <h2>知识库内容查看器</h2>
      <el-button 
        type="primary" 
        @click="loadCollections" 
        :loading="loading.collections"
        icon="Refresh"
      >
        刷新
      </el-button>
    </header>

    <div class="viewer-container">
      <!-- 左侧集合列表 -->
      <div class="collections-panel">
        <h3>集合列表</h3>
        <div class="collections-list">
          <el-skeleton 
            v-if="loading.collections" 
            :rows="4" 
            animated 
          />
          <div 
            v-else
            class="collection-item"
            v-for="collection in collections" 
            :key="collection"
            :class="{ active: selectedCollection === collection }"
            @click="selectCollection(collection)"
          >
            <el-icon><Collection /></el-icon>
            <span>{{ collection }}</span>
            <el-tag size="small" type="info">{{ collectionStats[collection]?.document_count || 0 }}</el-tag>
          </div>
          <div v-if="collections.length === 0 && !loading.collections" class="empty-state">
            <el-empty description="暂无集合" />
          </div>
        </div>
      </div>

      <!-- 右侧详情区域 -->
      <div class="details-panel">
        <div v-if="selectedCollection">
          <!-- 集合统计信息 -->
          <div class="stats-section">
            <h3>集合统计</h3>
            <el-card class="stats-card">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="集合名称">
                  {{ selectedCollection }}
                </el-descriptions-item>
                <el-descriptions-item label="文档数量">
                  {{ collectionStats[selectedCollection]?.document_count || 0 }}
                </el-descriptions-item>
                <el-descriptions-item label="主键字段">
                  {{ collectionStats[selectedCollection]?.primary_field || 'N/A' }}
                </el-descriptions-item>
                <el-descriptions-item label="描述">
                  {{ collectionStats[selectedCollection]?.description || 'N/A' }}
                </el-descriptions-item>
              </el-descriptions>
            </el-card>
          </div>

          <!-- 文档列表 -->
          <div class="documents-section">
            <h3>
              文档列表
              <el-tag type="info" size="small" effect="plain">
                共 {{ totalDocuments }} 条
              </el-tag>
            </h3>
            
            <el-skeleton 
              v-if="loading.documents" 
              :rows="6" 
              animated 
            />
            <div v-else>
              <el-table
                :data="documents"
                style="width: 100%"
                height="calc(100vh - 350px)"
                :default-sort="{ prop: 'id', order: 'ascending' }"
              >
                <el-table-column prop="id" label="ID" width="100" sortable />
                <el-table-column prop="metadata.source" label="文件名" width="200">
                  <template #default="{ row }">
                    {{ getFileName(row.metadata?.source) || '未知' }}
                  </template>
                </el-table-column>
                <el-table-column prop="content" label="内容摘要" min-width="300">
                  <template #default="{ row }">
                    <div class="content-preview" @click="showDocumentDetail(row)" style="cursor: pointer;">
                      {{ row.content.length > 200 ? row.content.substring(0, 200) + '...' : row.content }}
                      <el-tag v-if="row.content.length > 200" size="small" type="info" style="margin-left: 10px;">展开</el-tag>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column prop="full_length" label="长度" width="100" sortable />
                <el-table-column prop="fragment_count" label="片段数" width="100" sortable />
                <el-table-column label="操作" width="120">
                  <template #default="{ row }">
                    <el-button 
                      size="small" 
                      type="primary" 
                      text 
                      @click="showDocumentDetail(row)"
                    >
                      详情
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
              
              <!-- 分页 -->
              <div class="pagination-wrapper">
                <el-pagination
                v-model:current-page="currentPage"
                v-model:page-size="pageSize"
                :page-sizes="[10, 20, 50, 100]"
                layout="total, sizes, prev, pager, next, jumper"
                :total="totalCount"
                @size-change="handleSizeChange"
                @current-change="handleCurrentChange"
              />
              <div style="margin-top: 10px; font-size: 14px; color: #606266;">
                提示：表格中显示的是按来源分组的完整文档，总片段数为 {{ totalCount }}，当前页显示 {{ documents.length }} 个文档。
              </div>
              </div>
            </div>
          </div>
        </div>

        <div v-else class="welcome-panel">
          <el-empty description="请选择左侧的集合以查看内容" />
        </div>
      </div>
    </div>

    <!-- 文档详情弹窗 -->
    <el-dialog
      v-model="showDetailDialog"
      title="文档详情"
      width="60%"
      top="5vh"
    >
      <div class="document-detail">
        <h4>文档 ID: {{ currentDocument?.id }}</h4>
        <div class="detail-content">
          <h5>内容:</h5>
          <pre class="content-text">{{ currentDocument?.content }}</pre>
        </div>
        <div class="detail-metadata" v-if="currentDocument?.metadata">
          <h5>元数据:</h5>
          <pre class="metadata-text">{{ JSON.stringify(currentDocument.metadata, null, 2) }}</pre>
        </div>
      </div>
      <template #footer>
        <el-button @click="showDetailDialog = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { 
  Refresh, 
  Collection,
  Document
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

// API基础URL
const API_BASE = 'http://localhost:5000/api/vector'

// 状态管理
const collections = ref([])
const selectedCollection = ref('')
const collectionStats = ref({})
const documents = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const totalDocuments = ref(0)
const totalCount = ref(0)
const showDetailDialog = ref(false)
const currentDocument = ref(null)

// 加载状态
const loading = ref({
  collections: false,
  documents: false,
  delete: false
})

// 加载所有集合
const loadCollections = async () => {
  loading.value.collections = true
  try {
    const response = await axios.get(`${API_BASE}/collections`)
    if (response.data.success) {
      collections.value = response.data.collections || []
      // 加载每个集合的统计信息
      for (const collection of collections.value) {
        loadCollectionStats(collection)
      }
    } else {
      ElMessage.error(response.data.message || '获取集合列表失败')
    }
  } catch (error) {
    console.error('获取集合列表失败:', error)
    ElMessage.error('获取集合列表失败: ' + (error.response?.data?.message || error.message))
  } finally {
    loading.value.collections = false
  }
}

// 加载集合统计信息
const loadCollectionStats = async (collectionName) => {
  try {
    const response = await axios.get(`${API_BASE}/collections/${collectionName}/stats`)
    if (response.data.success) {
      collectionStats.value[collectionName] = response.data.stats
    }
  } catch (error) {
    console.error(`获取集合 ${collectionName} 统计信息失败:`, error)
  }
}

// 选择集合
const selectCollection = async (collection) => {
  selectedCollection.value = collection
  currentPage.value = 1
  await loadDocuments()
}

// 加载文档列表
const loadDocuments = async () => {
  if (!selectedCollection.value) return
  
  loading.value.documents = true
  try {
    const response = await axios.get(`${API_BASE}/collections/${selectedCollection.value}/documents`, {
      params: {
        page: currentPage.value,
        limit: pageSize.value
      }
    })
    
    if (response.data.success) {
      documents.value = response.data.documents || []
      totalDocuments.value = response.data.total_count || 0
      totalCount.value = response.data.total_count || 0
    } else {
      ElMessage.error(response.data.message || '获取文档列表失败')
    }
  } catch (error) {
    console.error('获取文档列表失败:', error)
    ElMessage.error('获取文档列表失败: ' + (error.response?.data?.message || error.message))
  } finally {
    loading.value.documents = false
  }
}

// 显示文档详情
const showDocumentDetail = (document) => {
  currentDocument.value = document
  showDetailDialog.value = true
}

// 分页处理
const handleSizeChange = (size) => {
  pageSize.value = size
  loadDocuments()
}

const handleCurrentChange = (page) => {
  currentPage.value = page
  loadDocuments()
}

// 删除文档
const deleteDocument = async (docId) => {
  if (!selectedCollection.value) {
    ElMessage.warning('请先选择一个集合')
    return
  }
  
  loading.value.delete = true
  try {
    const response = await axios.delete(
      `${API_BASE}/collections/${selectedCollection.value}/documents/${docId}`
    )
    
    if (response.data.success) {
      ElMessage.success(`成功删除文档 ${docId}: ${response.data.message}`)
      // 重新加载文档列表
      await loadDocuments()
    } else {
      ElMessage.error(response.data.message || '删除文档失败')
    }
  } catch (error) {
    console.error('删除文档失败:', error)
    ElMessage.error('删除文档失败: ' + (error.response?.data?.message || error.message))
  } finally {
    loading.value.delete = false
  }
}

// 获取文件名的辅助函数
const getFileName = (filePath) => {
  if (!filePath) return '未知'
  // 提取文件路径中的文件名部分
  return filePath.split('/').pop().split('\\').pop() || filePath
}

// 初始化
onMounted(() => {
  loadCollections()
})
</script>

<style scoped>
.knowledge-base-viewer {
  height: calc(100vh - 60px);
  display: flex;
  flex-direction: column;
  padding: 10px;
  box-sizing: border-box;
}

.viewer-header {
  padding: 15px 20px;
  background: #f5f7fa;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.viewer-header h2 {
  margin: 0;
  color: #303133;
  font-size: 18px;
}

.viewer-container {
  display: flex;
  flex: 1;
  overflow: hidden;
  border-radius: 6px;
  border: 1px solid #ebeef5;
}

.collections-panel {
  width: 250px;
  background: #fafafa;
  border-right: 1px solid #ebeef5;
  padding: 15px;
  overflow-y: auto;
}

.collections-panel h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #303133;
}

.collection-item {
  padding: 12px 15px;
  margin-bottom: 8px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.3s;
  background: #ffffff;
  border: 1px solid #ebeef5;
}

.collection-item:hover {
  background: #e6f7ff;
  border-color: #91d5ff;
}

.collection-item.active {
  background: #e6f7ff;
  border-color: #1890ff;
  color: #1890ff;
}

.collection-item .el-icon {
  color: #409eff;
}

.collection-item .el-tag {
  margin-left: auto;
}

.empty-state {
  text-align: center;
  padding: 40px 0;
}

.details-panel {
  flex: 1;
  padding: 15px;
  overflow-y: auto;
  background: #ffffff;
  display: flex;
  flex-direction: column;
}

.welcome-panel {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stats-section, .documents-section {
  margin-bottom: 25px;
}

.stats-section h3, .documents-section h3 {
  margin-top: 0;
  color: #303133;
}

.stats-card {
  margin-top: 15px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.view-controls {
  display: flex;
  align-items: center;
}

.content-preview {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.fragment-list {
  padding-left: 20px;
}

.fragment-item {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  margin-bottom: 10px;
  padding: 15px;
  background-color: #fafafa;
  cursor: pointer;
  transition: all 0.3s;
}

.fragment-item:hover {
  background-color: #f5f7fa;
  border-color: #dcdfe6;
}

.fragment-content {
  margin-bottom: 10px;
  line-height: 1.5;
}

.fragment-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.document-detail {
  max-height: 70vh;
  overflow-y: auto;
}

.document-detail h4, .document-detail h5 {
  color: #303133;
}

.detail-content, .detail-metadata {
  margin-bottom: 20px;
}

.content-text, .metadata-text {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 300px;
  overflow-y: auto;
}

.content-text {
  font-size: 14px;
  line-height: 1.5;
}
</style>