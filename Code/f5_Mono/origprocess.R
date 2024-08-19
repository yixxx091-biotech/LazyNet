library(Seurat)

metacomb=readRDS("./origdata.rds")

metacomb=NormalizeData(metacomb, normalization.method = "LogNormalize", scale.factor = 10000)
metacomb=FindVariableFeatures(metacomb, selection.method = "vst", nfeatures = 0.05*nrow(metacomb))
all.genes=rownames(metacomb)
metacomb=ScaleData(metacomb, features = all.genes)
metacomb=RunPCA(metacomb, features = VariableFeatures(object = metacomb))
metacomb=RunUMAP(metacomb,dims=1:15)
metacomb=FindNeighbors(metacomb, dims = 1:15)
metacomb=FindClusters(metacomb, resolution = 0.2)
metacomb=subset(metacomb, idents = 5, invert= T)
metacomb=FindNeighbors(metacomb, dims = 1:15)
metacomb=FindClusters(metacomb, resolution = 0.3) 
metacomb=subset(metacomb, idents = 7, invert= T)
DimPlot(metacomb,reduction = "umap")

mi=Idents(metacomb)
mi=paste0("M",mi)
Idents(metacomb)=mi

bigdata=GetAssayData(metacomb, slot = "data")
bigdata=bigdata[which(rownames(bigdata)%in%VariableFeatures(metacomb)),]

pick_random_cell <- function(cluster_pool, all_cells, cluster_labels) {
  cluster_points <- all_cells[cluster_labels %in% cluster_pool]
  if (length(cluster_points) > 0) {
    picked_cell <- sample(cluster_points, 1)
    return(picked_cell)
  }
  return(NULL)
}

clusters <- Idents(metacomb)
all_cells <- colnames(metacomb)

clusters_to_pick <- list("M0", "M2", "M4", "M6", "M5", "M3", "M1")

output_list <- list()
for (i in 1:2000) {
  picked_cells <- character()
  for (cluster_or_pool in clusters_to_pick) {
    picked_cell <- pick_random_cell(cluster_or_pool, all_cells, clusters)
    if (!is.null(picked_cell)) {
      picked_cells <- c(picked_cells, picked_cell)
    }
  }
  if (length(picked_cells) > 0) {
    pickid <- match(picked_cells, colnames(bigdata))
    cell_data <- bigdata[, pickid, drop = FALSE]
    output_list[[i]] <- cell_data
  }
}
output_list=as.data.frame(output_list)
dim(output_list)
write.csv(output_list,"../trainset_bigc_7_053024.csv")
