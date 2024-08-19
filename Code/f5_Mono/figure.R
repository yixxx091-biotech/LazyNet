library(Seurat)

metacomb=readRDS("./origdata.rds")
metacomb=NormalizeData(metacomb, normalization.method = "LogNormalize", scale.factor = 10000)
metacomb=FindVariableFeatures(metacomb, selection.method = "vst", nfeatures = 0.05*nrow(metacomb))
all.genes=rownames(metacomb)
metacomb=ScaleData(metacomb, features = all.genes)
metacomb=RunPCA(metacomb, features = VariableFeatures(object = metacomb))
gene717=rownames(Loadings(metacomb))
metacomb=RunUMAP(metacomb,dims=1:15)
metacomb=FindNeighbors(metacomb, dims = 1:15)
metacomb=FindClusters(metacomb, resolution = 0.2)
metacomb=subset(metacomb, idents = 5, invert= T)
metacomb=FindNeighbors(metacomb, dims = 1:15)
metacomb=FindClusters(metacomb, resolution = 0.3) 
metacomb=subset(metacomb, idents = 7, invert= T)
Idents(metacomb,cells=which(Idents(metacomb)==1))="O7"
Idents(metacomb,cells=which(Idents(metacomb)==3))="O6"
Idents(metacomb,cells=which(Idents(metacomb)==5))="O5"
Idents(metacomb,cells=which(Idents(metacomb)==6))="O4"
Idents(metacomb,cells=which(Idents(metacomb)==4))="O3"
Idents(metacomb,cells=which(Idents(metacomb)==2))="O2"
Idents(metacomb,cells=which(Idents(metacomb)==0))="O1"
metacomb=metacomb[which(rownames(metacomb)%in%gene717),]
DimPlot(metacomb, reduction = "umap")

trainset=read.table("./trainset_Seurat.csv",header = F, sep = ",")
trainset=trainset[,-1]
rownames(trainset)=gene717
cols_with_inf <- apply(trainset, 2, function(x) any(is.infinite(x)))
trainset <- trainset[, !cols_with_inf]
trainset[trainset < 0] <- 0
trainset=CreateSeuratObject(counts = trainset, project = "trainset", min.cells = 3, min.features = 200)
trainset=NormalizeData(trainset, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes=rownames(trainset)
trainset=ScaleData(trainset, features = all.genes)
trainset=RunPCA(trainset, features = all.genes)
trainset=RunUMAP(trainset,dims=1:14)
trainset=FindNeighbors(trainset, dims = 1:14)
trainset=FindClusters(trainset, resolution = 0.1)
trainset=subset(trainset,idents = 4,invert=T)
Idents(trainset,cells=which(Idents(trainset)==0))="T4"
Idents(trainset,cells=which(Idents(trainset)==1))="T3"
Idents(trainset,cells=which(Idents(trainset)==2))="T2"
Idents(trainset,cells=which(Idents(trainset)==3))="T1"
DimPlot(trainset,reduction = "umap")

predat=read.table("./bigc_trainset140_053024_clamp_quad_pred_700dp.csv",header = F, sep = ",")
predat=predat[,-1]
rownames(predat)=rownames(Loadings(metacomb))
cols_with_inf <- apply(predat, 2, function(x) any(is.infinite(x)))
predat <- predat[, !cols_with_inf]
predat[predat < 0] <- 0
predat=CreateSeuratObject(counts = predat, project = "predicted", min.cells = 3, min.features = 200)
predat=NormalizeData(predat, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes=rownames(predat)
predat=ScaleData(predat, features = all.genes)
predat=RunPCA(predat, features = all.genes)
predat=RunUMAP(predat,dims=1:14)
predat=FindNeighbors(predat, dims = 1:14)
predat=FindClusters(predat, resolution = 0.2)
predat=subset(predat, idents = c(0,1,2,3,4,5))
predat=RunUMAP(predat,dims=1:14)
Idents(predat,cells=which(Idents(predat)==5))="P6"
Idents(predat,cells=which(Idents(predat)==4))="P5"
Idents(predat,cells=which(Idents(predat)==3))="P4"
Idents(predat,cells=which(Idents(predat)==2))="P3"
Idents(predat,cells=which(Idents(predat)==1))="P2"
Idents(predat,cells=which(Idents(predat)==0))="P1"
DimPlot(predat, reduction = "umap")

fig=merge(metacomb, y = trainset, add.cell.ids = c("orig","trainset"), project = "fig")
fig=NormalizeData(fig, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes=rownames(fig)
fig=ScaleData(fig, features = all.genes)
fig=RunPCA(fig, features = all.genes)
fig=RunUMAP(fig,dims=1:14)
png("../pub080624_model6_trainset140_quad_700pt_oVSt_s20.png", width = 6000, height = 6000, res = 600)
dplot=DimPlot(fig,reduction = "umap")
print(dplot)
dev.off()

fig=merge(predat, y = trainset, add.cell.ids = c("pred","trainset"), project = "fig")
fig=NormalizeData(fig, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes=rownames(fig)
fig=ScaleData(fig, features = all.genes)
fig=RunPCA(fig, features = all.genes)
fig=RunUMAP(fig,dims=1:14)
png("../pub080624_model6_trainset140_quad_700pt_pVSt_s20.png", width = 6000, height = 6000, res = 600)
dplot=DimPlot(fig,reduction = "umap")
print(dplot)
dev.off()

fig=merge(predat, y = metacomb, add.cell.ids = c("pred","orig"), project = "fig")
fig=NormalizeData(fig, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes=rownames(fig)
fig=ScaleData(fig, features = all.genes)
fig=RunPCA(fig, features = all.genes)
fig=RunUMAP(fig,dims=1:14)
png("../pub080624_model6_trainset140_quad_700pt_oVSp_s20.png", width = 6000, height = 6000, res = 600)
dplot=DimPlot(fig,reduction = "umap")
print(dplot)
dev.off()

png("./pub080624_model6_orig_heat.png", width = 4000, height = 20000, res = 600)
dplot=DoHeatmap(metacomb, features = gene717)
print(dplot)
dev.off()
png("./pub080624_model6_trainset_heat.png", width = 4000, height = 20000, res = 600)
dplot=DoHeatmap(trainset, features = gene717)
print(dplot)
dev.off()
png("./pub080624_model6_pred_heat.png", width = 4000, height = 20000, res = 600)
dplot=DoHeatmap(predat, features = gene717, disp.min = -1, disp.max = 1)
print(dplot)
dev.off()

intgene=FindMarkers(predat, ident.1=c("P5", "P6"), ident.2 = c("P1","P2","P3","P4"),logfc.threshold = 1)
intgene=intgene[order(intgene$avg_log2FC,decreasing = T),]
write.table(intgene,"./P56vs1234genes.txt")

test=c("Ccl2","Ccl4","S100a8","S100a9","Tnf","Rsad2","Plac8","Stat1","Klf4","Irf7")
png("../pub080624_model6_pred_dot.png", width = 6000, height = 3000, res = 600)
dplot=DotPlot(predat,features = test,cols = "RdYlBu",scale.min = -40)
print(dplot)
dev.off()
png("../pub080624_model6_orig_dot.png", width = 6000, height = 3000, res = 600)
dplot=DotPlot(metacomb,features = test,cols = "RdYlBu",scale.min = -40)
print(dplot)
dev.off()