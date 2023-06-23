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

        for i, (section, pagenum) in enumerate(split_text(page_map)):
            yield {
                "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{filename}-{i}"),
                "content": section,
                "category": "",
                "sourcepage": self.blob_name_from_file_page(filename, pagenum),
                "sourcefile": filename
            }

        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=MAX_SECTION_LENGTH,
        #     chunk_overlap=SECTION_OVERLAP,
        # )
        #
        # all_text = "".join(p[2] for p in page_map)
        #
        # text_sections = splitter.create_documents([all_text])
        #
        # sections = [
        #     {
        #         "id": re.sub("[^0-9a-zA-Z_-]", "_", f"{filename}-{i}"),
        #         "content": section.page_content,
        #         "category": "",
        #         "sourcepage": self.blob_name_from_file_page(
        #             filename
        #         ),  # XXX: Page number isn't working
        #         "sourcefile": filename,
        #     }
        #     for i, section in enumerate(text_sections)
        # ]
        # return sections

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


def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[
                end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word  # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[
            start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))
