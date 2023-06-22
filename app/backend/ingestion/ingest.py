import os
import re
import io
import glob
from azure.search.documents.indexes.models import *

from langchain.text_splitter import RecursiveCharacterTextSplitter

from pypdf import PdfReader, PdfWriter

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100


class Ingest:
    def __init__(
        self, AZURE_SEARCH_INDEX, search_index_client, blob_container, search_client
    ) -> None:
        self.index = AZURE_SEARCH_INDEX
        self.search_index_client = search_index_client
        self.blob_container = blob_container
        self.search_client = search_client

    def blob_name_from_file_page(self, filename, page=0):
        if os.path.splitext(filename)[1].lower() == ".pdf":
            return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
        else:
            return os.path.basename(filename)

    def create_search_index(self):
        print(f"Ensuring search index {self.index} exists")
        if self.index not in self.search_index_client.list_index_names():
            my_index = SearchIndex(
                name=self.index,
                fields=[
                    SimpleField(name="id", type="Edm.String", key=True),
                    SearchableField(
                        name="content", type="Edm.String", analyzer_name="en.microsoft"
                    ),
                    SimpleField(
                        name="category",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                    ),
                    SimpleField(
                        name="sourcepage",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                    ),
                    SimpleField(
                        name="sourcefile",
                        type="Edm.String",
                        filterable=True,
                        facetable=True,
                    ),
                ],
                semantic_settings=SemanticSettings(
                    configurations=[
                        SemanticConfiguration(
                            name="default",
                            prioritized_fields=PrioritizedFields(
                                title_field=None,
                                prioritized_content_fields=[
                                    SemanticField(field_name="content")
                                ],
                            ),
                        )
                    ]
                ),
            )

            print(f"Creating {self.index} search index")
            self.search_index_client.create_index(my_index)
        else:
            print(f"Search index {self.index} already exists")

    def upload_blobs(self, filename):
        if not self.blob_container.exists():
            self.blob_container.create_container()

        if os.path.splitext(filename)[1].lower() == ".pdf":
            reader = PdfReader(filename)
            pages = reader.pages
            for i in range(len(pages)):
                blob_name = self.blob_name_from_file_page(filename, i)
                print(f"\tUploading blob for page {i} -> {blob_name}")
                f = io.BytesIO()
                writer = PdfWriter()
                writer.add_page(pages[i])
                writer.write(f)
                f.seek(0)
                self.blob_container.upload_blob(blob_name, f, overwrite=True)
        else:
            blob_name = self.blob_name_from_file_page(filename)
            with open(filename, "rb") as data:
                self.blob_container.upload_blob(blob_name, data, overwrite=True)

    def get_document_text(self, filename):
        offset = 0
        page_map = []

        if os.path.splitext(filename)[1].lower() == ".pdf":
            reader = PdfReader(filename)
            pages = reader.pages
            for page_num, p in enumerate(pages):
                page_text = p.extract_text()
                page_map.append((page_num, offset, page_text))
                offset += len(page_text)

        return page_map

    def create_sections(self, filename, page_map):
        print(f"Splitting '{filename}' into sections")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_SECTION_LENGTH,
            chunk_overlap=SECTION_OVERLAP,
        )

        all_text = "".join(p[2] for p in page_map)
        text_sections = splitter.create_documents([all_text])

        sections = [
            {
                "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{filename}-{i}"),
                "content": section.page_content,
                "category": "",
                "sourcepage": self.blob_name_from_file_page(
                    filename
                ),  # XXX: Page number isn't working
                "sourcefile": filename,
            }
            for i, section in enumerate(text_sections)
        ]
        return sections

    def index_sections(self, filename, sections):
        print(f"Indexing sections from '{filename}' into search index '{self.index}'")
        i = 0
        batch = []

        for s in sections:
            batch.append(s)
            i += 1
            if i % 1000 == 0:
                results = self.search_client.upload_documents(documents=batch)
                succeeded = sum([1 for r in results if r.succeeded])
                print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
                batch = []

        if len(batch) > 0:
            results = self.search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

    def run(self):
        self.create_search_index()

        print(f"Processing files...")
        for filename in glob.glob("data/*"):
            print(f"Processing '{filename}'")
            self.upload_blobs(filename)
            page_map = self.get_document_text(filename)
            sections = self.create_sections(os.path.basename(filename), page_map)
            self.index_sections(os.path.basename(filename), sections)
