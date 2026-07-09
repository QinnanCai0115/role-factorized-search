import unittest
from verl.utils.reward_score.rag_2 import extract_titles_and_texts

class TestExtractTitlesAndTexts(unittest.TestCase):
    def test_empty_input(self):
        """Test empty input string"""
        result = extract_titles_and_texts("")
        self.assertEqual(result, [])

    def test_no_information_blocks(self):
        """Test string with no information blocks"""
        result = extract_titles_and_texts("Some random text")
        self.assertEqual(result, [])

    def test_no_important_info(self):
        """Test information block without important_info tag"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [])

    def test_single_doc_important_info(self):
        """Test single document in important_info"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>1</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [("Title 1", "Content 1")])

    def test_multiple_docs_important_info(self):
        """Test multiple documents in important_info"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>1,2</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [("Title 1", "Content 1"), ("Title 2", "Content 2")])

    def test_different_important_info_formats(self):
        """Test different formats of important_info"""
        test_cases = [
            ("<important_info>1</important_info>", [("Title 1", "Content 1")]),
            ("<important_info>Doc1</important_info>", [("Title 1", "Content 1")]),
            ("<important_info>[1]</important_info>", [("Title 1", "Content 1")]),
            ("<important_info>Doc 1</important_info>", [("Title 1", "Content 1")]),
            ("<important_info>1,2,3</important_info>", [("Title 1", "Content 1"), ("Title 2", "Content 2"), ("Title 3", "Content 3")]),
            ("<important_info>[1,2,3]</important_info>", [("Title 1", "Content 1"), ("Title 2", "Content 2"), ("Title 3", "Content 3")]),
            ("<important_info>Doc1,Doc2,Doc3</important_info>", [("Title 1", "Content 1"), ("Title 2", "Content 2"), ("Title 3", "Content 3")]),
            ("<important_info>Doc 1, Doc 2, Doc 3</important_info>", [("Title 1", "Content 1"), ("Title 2", "Content 2"), ("Title 3", "Content 3")]),
        ]
        
        base_info = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        """
        
        for important_info, expected in test_cases:
            input_str = base_info + important_info
            result = extract_titles_and_texts(input_str)
            self.assertEqual(result, expected, f"Failed for format: {important_info}")

    def test_invalid_doc_ids(self):
        """Test important_info with invalid document IDs"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>4,5,6</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [])

    def test_multiple_information_blocks(self):
        """Test multiple information blocks with different important_info tags"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>1,2</important_info>
        <information>
        Doc 1(Title: "Title 1") Content 4
        Doc 2(Title: "Title 2") Content 5
        Doc 3(Title: "Title 3") Content 6
        </information>
        <important_info>2,3</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [
            ("Title 1", "Content 1"),  # Doc 1 from first block
            ("Title 2", "Content 2"),  # Doc 2 from first block
            ("Title 2", "Content 5"),  # Doc 2 from second block
            ("Title 3", "Content 6")   # Doc 3 from second block
        ])

    def test_duplicate_documents(self):
        """Test handling of duplicate documents"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        </information>
        <important_info>1,2</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [("Title 1", "Content 1"), ("Title 2", "Content 2")])

    def test_malformed_input(self):
        """Test handling of malformed input"""
        test_cases = [
            "<information>Doc 1(Title: Malformed content",
            "<important_info>1</important_info>",
            "<information>Doc 1(Title: Title 1) Content 1</information><important_info>",
            "<information>Doc 1(Title: Title 1) Content 1</information><important_info>invalid</important_info>",
        ]
        
        for input_str in test_cases:
            result = extract_titles_and_texts(input_str)
            self.assertEqual(result, [], f"Failed for malformed input: {input_str}")

    def test_mixed_case_important_info(self):
        """Test important_info with mixed case document references"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>Doc1, doc2, DOC3</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [("Title 1", "Content 1"), ("Title 2", "Content 2"), ("Title 3", "Content 3")])

    # def test_real_world_scenario(self):
    #     """Test a real-world scenario with multiple information blocks and important_info tags"""
    #     input_str = """
    #     <think>
    #     I will start by formulating an initial search query to find out the holiday weekend related to the Purdue All-American Marching Band's hosting event.
    #     </think>
    #     <query>
    #     {
    #         "query": "Holiday weekend for Purdue All-American Marching Band host event"
    #     }
    #     </query>
    #     <information>
    #     Doc 1(Title: "Purdue All-American Marching Band") every year. Previous years membership does not guarantee a position in the band. Auditions are composed of a music audition followed by a week of marching auditions prior to the first week of classes. Student Leaders for each instrument section are selected by band staff to aid in the teaching of Purdue band marching fundamentals including the Big Ten conference's ubiquitous chair step. Purdue All-American Marching Band The Purdue "All-American" Marching Band (or AAMB) is the marching band of Purdue University and the main source of auxiliary entertainment for Purdue Boilermakers football games. The AAMB is also the host band
    #     Doc 2(Title: "Purdue All-American Marching Band") Purdue All-American Marching Band The Purdue "All-American" Marching Band (or AAMB) is the marching band of Purdue University and the main source of auxiliary entertainment for Purdue Boilermakers football games. The AAMB is also the host band of the Indianapolis 500 race, having held the position since 1919. The band has grown from an original 5 members to 389 members. The three most distinctive features of the AAMB are the World's Largest Drum, the Purdue Golden Girl featured twirler, and the "Block P," the first marching band field formation created in 1907. In 1886 the Purdue Student Army Training Corps
    #     Doc 3(Title: "Purdue All-American Marching Band") in many countries including Canada, Holland, Germany, Iceland, Japan, Colombia, and Venezuela during the 1960s and 1970s. This tradition of international travel continues today, with other notable trips including an invitation from the Chinese government to perform in international cultural activities that lead up to the 2008 Beijing Summer Olympics. In both 2012 and 2018 the Band also traveled to Ireland for the Saint Patrick's Day Parade in Dublin. In 2015, the AAMB traveled to Medellin, Colombia for the Feria de las Flores. In 1995 the "All-American" Marching Band was the recipient of the Sudler Trophy, the most prestigious award
    #     </information>
    #     <important_info>1, 2</important_info>
    #     <search_complete>False</search_complete>
    #     <query>
    #     {
    #         "query": "Holiday weekend for the Indianapolis 500 race"
    #     }
    #     </query>
    #     <information>
    #     Doc 1(Title: "Indianapolis 500 traditions") are announced. The race has always been scheduled in conjunction with Memorial Day. Through 1970, the race was held on Memorial Day proper (May 30), regardless of the day of the week, unless it fell on Sunday. In those cases it was scheduled for Monday May 31. After the Uniform Monday Holiday Act took effect in 1971, the race was scheduled as part of the three-day Memorial Day weekend - either the Saturday, the Sunday, or the Monday of the holiday weekend. Since 1974, the race has been scheduled for the Sunday of Memorial Day weekend, and has been held
    #     Doc 2(Title: "Indianapolis 500") Indianapolis 500 The Indianapolis 500-Mile Race (better known as the Indianapolis 500) is an automobile race held annually at Indianapolis Motor Speedway (IMS) in Speedway, Indiana, United States, an enclave suburb of Indianapolis, Indiana. The event is held over Memorial Day weekend in late May. It is contested as part of the Verizon INDYCAR Series, the top level of American Championship Car racing, an open-wheel open-cockpit formula colloquially known as "Indy Car Racing". The name of the race is often shortened to Indy 500, and the track itself is nicknamed "the Brickyard", as the racing surfacing was paved in brick
    #     Doc 3(Title: "Indianapolis 500") this contract; the existing blackout policy is expected to continue. Indianapolis 500 The Indianapolis 500-Mile Race (better known as the Indianapolis 500) is an automobile race held annually at Indianapolis Motor Speedway (IMS) in Speedway, Indiana, United States, an enclave suburb of Indianapolis, Indiana. The event is held over Memorial Day weekend in late May. It is contested as part of the Verizon INDYCAR Series, the top level of American Championship Car racing, an open-wheel open-cockpit formula colloquially known as "Indy Car Racing". The name of the race is often shortened to Indy 500, and the track itself is nicknamed
    #     </information>
    #     <important_info>1, 2</important_info>
    #     <search_complete>False</search_complete>
    #     """
        
    #     # Extract documents from the first information block
    #     result = extract_titles_and_texts(input_str)
    #     expected = [
    #         ("Purdue All-American Marching Band", "every year. Previous years membership does not guarantee a position in the band. Auditions are composed of a music audition followed by a week of marching auditions prior to the first week of classes. Student Leaders for each instrument section are selected by band staff to aid in the teaching of Purdue band marching fundamentals including the Big Ten conference's ubiquitous chair step. Purdue All-American Marching Band The Purdue \"All-American\" Marching Band (or AAMB) is the marching band of Purdue University and the main source of auxiliary entertainment for Purdue Boilermakers football games. The AAMB is also the host band"),
    #         ("Purdue All-American Marching Band", "Purdue All-American Marching Band The Purdue \"All-American\" Marching Band (or AAMB) is the marching band of Purdue University and the main source of auxiliary entertainment for Purdue Boilermakers football games. The AAMB is also the host band of the Indianapolis 500 race, having held the position since 1919. The band has grown from an original 5 members to 389 members. The three most distinctive features of the AAMB are the World's Largest Drum, the Purdue Golden Girl featured twirler, and the \"Block P,\" the first marching band field formation created in 1907. In 1886 the Purdue Student Army Training Corps")
    #     ]
    #     self.assertEqual(result, expected)

    def test_multiple_important_info_tags(self):
        """Test multiple important_info tags with different document selections"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1
        Doc 2(Title: "Title 2") Content 2
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>1</important_info>
        <think>Some thinking here</think>
        <important_info>2</important_info>
        <query>Some query here</query>
        <important_info>3</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [("Title 1", "Content 1")])

    def test_nested_tags(self):
        """Test information blocks with nested tags"""
        input_str = """
        <information>
        Doc 1(Title: "Title 1") Content 1 <think>Some thinking</think>
        Doc 2(Title: "Title 2") Content 2 <query>Some query</query>
        Doc 3(Title: "Title 3") Content 3
        </information>
        <important_info>1, 2, 3</important_info>
        """
        result = extract_titles_and_texts(input_str)
        self.assertEqual(result, [
            ("Title 1", "Content 1 <think>Some thinking</think>"),
            ("Title 2", "Content 2 <query>Some query</query>"),
            ("Title 3", "Content 3")
        ])

if __name__ == '__main__':
    unittest.main() 